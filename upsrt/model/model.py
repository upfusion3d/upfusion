import math
import torch
import torch.nn as nn

from upsrt.model.resnet import ResNetConv
from upsrt.model.transformer import (
    TransformerEncoder, TransformerEncoderBlock,
    TransformerDecoder, TransformerDecoderBlock
)

from upsrt.model.utils import plucker_dist, transform_rays
from upsrt.renderer.rays import (
    get_grid_rays, get_patch_rays,
    get_plucker_parameterization,
    get_random_query_pixel_rays,
    positional_encoding, get_grid_rays_gpu
)

from pytorch3d.renderer.cameras import PerspectiveCameras
from upsrt.utils.id_encoding import create_patch_id_encoding, create_camera_id_encoding

ATTN_TYPE = "xformers"


class SceneEncoder(nn.Module):
    """
    Takes set of patch-wise image and ray features as input and computes a set latent encoding for the scene
    """
    def __init__(self, cfg):
        super(SceneEncoder, self).__init__()

        # Transformer architecture params
        self.transformer_dim = cfg.transformer_dim
        self.encoder_hidden_activation = 'gelu'
        self.encoder_n_attention_heads = 12
        self.encoder_num_layers = cfg.num_encoder_layers

        self.transformer_encoder = TransformerEncoder(
            encoder_layer = TransformerEncoderBlock(
                attn_type=ATTN_TYPE, d_model=self.transformer_dim,
                nhead=self.encoder_n_attention_heads,
                activation=self.encoder_hidden_activation
            ),
            num_layers = self.encoder_num_layers
        )

    def forward(self, scene_features):
        """
        Args:
            scene_features: (b, n_inp, patch, transformer_dim)
            src_mask(torch.Tensor): FloatTensor (additive mask) of shape (b * n_heads, n_inp * patch, n_inp * patch)
        Returns:
            torch.Tensor: Tensor of shape (n_inp*patch, b, d_model) representing scene latent encoding
        """
        b, n_inp, n_patch, _ = scene_features.shape
        encoder_input = torch.reshape(scene_features, (b, n_inp * n_patch, self.transformer_dim))  # (b, n_inp*patch, d_model)
        scene_encoding = self.transformer_encoder(encoder_input)  # (b, n_inp*patch, d_model)

        return scene_encoding


class RayDecoder(nn.Module):
    """
    Decodes color value for each query pixel ray using a set latent encoding
    """
    def __init__(self, cfg):
        super(RayDecoder, self).__init__()

        # Transformer architecture params
        self.transformer_dim = cfg.transformer_dim
        self.decoder_hidden_activation = 'gelu'
        self.decoder_n_attention_heads = 12
        self.decoder_num_layers = cfg.num_decoder_layers

        self.transformer_decoder = TransformerDecoder(
            decoder_layer = TransformerDecoderBlock(
                attn_type=ATTN_TYPE, d_model=self.transformer_dim,
                nhead=self.decoder_n_attention_heads,
                activation=self.decoder_hidden_activation
            ),
            num_layers = self.decoder_num_layers
        )

        self.rgb_mlp = nn.Sequential(
            nn.Linear(self.transformer_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )

    def forward(self, query_pixel_rays, scene_encoding, return_type="pred_rgb"):
        """
        Args:
            query_pixel_rays: (b, num_pixel_queries, transformer_dim)
            scene_encoding: (n_inp * patch, b, transformer_dim)
            memory_mask: Tensor of shape (b * n_heads, num_queries, n_inp * patch)
        Returns:
            torch.Tensor: Tensor of shape (b, num_pixel_queries, 3) representing rgb value for each input ray
        """
        # Decode query rays using scene latent representation
        pred_embed = self.transformer_decoder(
            query_pixel_rays, scene_encoding
        )  # (b, num_pixel_queries, d_model)

        if return_type == "pre_rgb_mlp_features":
            output = pred_embed

        elif return_type == "pred_rgb":
            # Predict pixel rgb color values
            output = self.rgb_mlp(pred_embed)  # (b, num_pixel_queries, 3), in range [0, 1]

        else:
            raise ValueError(f"Invalid choice of return_type: {return_type}")

        return output


class UpSRT(nn.Module):
    """
    A "scene" represents a novel scene/object in 3D, and our input consists of multiple sparse views
    (num_input_views) corresponding to that scene. Aim is to form a scene embedding for input sparse
    view images and patch rays corresponding to these images. We pass the input images to a pre-trained
    feature extractor (could be anything, ViT or ResNet) to obtain patch embeddings of shape
    (num_input_views*P, D1). We also encode corresponding patch rays (corresponding to center pixel
    of each patch) of shape (num_input_views*P, D2). We concatenate the image and ray embeddings
    (num_input_views*P, D1+D2) and pass them to a transformer encoder to generate a scene encoding of
    dimensions - (num_input_views*P, D).

    The scene encoding from the transformer encoder is fed to another transformer decoder along with
    per-pixel query rays from a novel view point to generate novel view pixel values. We will then
    take a reconstruction loss between the predicted pixels and gt pixels.

    """

    def __init__(self, cfg):
        super(UpSRT, self).__init__()

        self.num_pixel_queries = cfg.num_pixel_queries

        # Image patch feature extractor
        self.image_feature_dim = cfg.feature_extractor.image_feature_dim
        self.num_patches_x = cfg.feature_extractor.num_patches_x
        self.num_patches_y = cfg.feature_extractor.num_patches_y

        # Ray positional encoding args
        self.num_freqs = cfg.ray.num_freqs
        self.start_freq = cfg.ray.start_freq
        self.parameterize = cfg.ray.parameterize
        self.harmonic_embedding_dim = 2 * self.num_freqs * 6
        self.view_space = cfg.ray.view_space

        # Transformer encoder and decoder
        self.transformer_dim = cfg.transformer_dim
        self.scene_encoder = SceneEncoder(cfg.scene_encoder)
        self.ray_decoder = RayDecoder(cfg.ray_decoder)

        # self.linear_img_features = nn.Linear(self.image_feature_dim, self.transformer_dim)

        self.linear_scene = nn.Linear(
            self.image_feature_dim + self.harmonic_embedding_dim + 2*self.num_freqs + 2*self.num_freqs,
            self.transformer_dim
        )
        self.linear_query_pixel_rays = nn.Linear(self.harmonic_embedding_dim, self.transformer_dim)

        # stddev = 1.0 / math.sqrt(self.transformer_dim)
        # self.first_camera_enc = nn.Parameter(
        #     data = torch.randn((1, 1, 1, self.transformer_dim)) * stddev
        # )
        # self.other_camera_enc = nn.Parameter(
        #     data = torch.randn((1, 1, 1, self.transformer_dim)) * stddev
        # )
        # self.patch_enc = nn.Parameter(
        #     data = torch.randn((1, 1, self.num_patches_x*self.num_patches_y, self.transformer_dim)) * stddev
        # )

    def forward(self, dino_features, input_cameras, query_pixel_rays, device, return_type="pred_rgb"):
        """
        Args:
            dino_features: (b, n, t, d)
            input_cameras: (list of list of cameras; list shape (B, num_input_views)).
            query_pixel_rays: (B, num_pixel_queries, 6) - note: (origin, direction) representation

        Returns:
            torch.Tensor: Predicted pixel values corresponding to query_pixel_rays of shape (B, num_pixel_queries, 3).
        """
        n_views = dino_features.shape[1]

        # Scene latent representation
        scene_encoding = self.encode(device, dino_features, input_cameras)  # (n_inp * patch, b, transformer_dim)

        pred_pixels = self.decode(
            scene_encoding, query_pixel_rays, input_cameras,
            device, return_type
        )
        return pred_pixels

    def encode(self, device, dino_features, input_cameras):
        """
        Args:
            dino_features: (b, n, t, d)
            input_cameras: Input cameras corresponding to each provided view for the batch (list of list cameras; list shape (B, num_input_views)).

        Returns:
            torch.Tensor: Predicted pixel values corresponding to query_pixel_rays of shape (B, num_pixel_queries, 3).

        """
        n_views = dino_features.shape[1]

        identity_cameras = self.create_cameras_with_identity_extrinsics(input_cameras)
        input_patch_rays = get_patch_rays(
            identity_cameras, num_patches_x=self.num_patches_x,
            num_patches_y=self.num_patches_y, device=device
        )

        # Convert to plucker and convert to harmonics embeddings
        input_patch_rays = positional_encoding(
            input_patch_rays, n_freqs=self.num_freqs,
            parameterize=self.parameterize, start_freq=self.start_freq
        )  # (b, n_inp, patch, pos_embedding_dim)

        patch_id_encoding = create_patch_id_encoding(
            dino_features.shape, num_patches=self.num_patches_x*self.num_patches_y,
            n_freqs=self.num_freqs, start_freq=self.start_freq
        ).to(device)

        camera_id_encoding = create_camera_id_encoding(
            dino_features.shape, num_patches=self.num_patches_x*self.num_patches_y,
            n_freqs=self.num_freqs, start_freq=self.start_freq
        ).to(device)

        # Concatenate input patch ray embeddings to image patch features
        scene_features = torch.cat(
            (dino_features, input_patch_rays, patch_id_encoding, camera_id_encoding), dim=-1
        ) # (b, n_inp, patch, img_feature_dim + pos_embedding_dim + 2*self.num_freqs + 2*self.num_freqs)

        # Project scene features to transformer dimensions
        scene_features = self.linear_scene(scene_features)  # (b, n_inp, patch, transformer_dim)

        # Scene latent representation
        scene_encoding = self.scene_encoder(scene_features)  # (b, n_inp * patch, transformer_dim)

        return scene_encoding

    def get_set_latent_representation(self, dino_features, input_cameras):

        scene_encoding = self.encode(
            device=dino_features.device, dino_features=dino_features,
            input_cameras=input_cameras
        )
        return scene_encoding

    def decode(
        self, scene_encoding, query_pixel_rays,
        input_cameras, device, return_type="pred_rgb"
    ):

        # Convert query rays to view space if required.
        query_pixel_rays = self.convert_query_rays_to_view_space(
            device, input_cameras, query_pixel_rays
        )

        # Encode and project query rays to transformer dimensions
        query_pixel_rays = positional_encoding(
            query_pixel_rays, n_freqs=self.num_freqs,
            parameterize=self.parameterize, start_freq=self.start_freq
        )  # (b, num_pixel_queries, pos_embedding_dim)

        query_pixel_rays = self.linear_query_pixel_rays(query_pixel_rays)  # (b, num_pixel_queries, transformer_dim)

        pred_pixels = self.ray_decoder(
            query_pixel_rays, scene_encoding, return_type = return_type
        )
        return pred_pixels

    def get_query_rays(self, query_cameras, image_size=None, query_ray_filter=None):

        if not self.training:
            raise RuntimeError("This function is only to be used during training.")

        return get_random_query_pixel_rays(
            query_cameras, num_pixel_queries=self.num_pixel_queries, query_ray_filter=query_ray_filter,
            min_x=1, max_x=-1, min_y=1, max_y=-1,
            return_xys=True, device='cpu'
        )

    def infer(self, dino_features, input_cameras, query_cameras, image_size=None):
        """Infers model for a given set of input views and the query view. Predicts the pixel values for all pixels (H*W) given the query view.
        Args:
            dino_features: (b, n, t, d)
            input_cameras(list[pytorch3d.renderer.cameras.CamerasBase]): List of Pytorch3D cameras of length (n_cameras,) corresponding to the
                                                                        input views.
            query_cameras(list[pytorch3d.renderer.cameras.CamerasBase]): List of Pytorch3D cameras of length (n_query_cameras,) corresponding to the
                                                                        query views.
            image_size(tuple[int, int]): Size of the image in pixels (height, width).

        Returns:
            torch.Tensor: Tensor of shape (n_query_cameras, H*W, 3) containing the predicted
                        pixel values for each pixel in each query view.
        """
        assert not self.training, "Set model.eval() before calling infer"
        with torch.no_grad():
            pred_pixels, _ = self.get_query_features(
                dino_features = dino_features, input_cameras = input_cameras,
                query_cameras = query_cameras, image_size = image_size,
                decoder_return_type = "pred_rgb"
            ) # (n_query_cameras, H*W, 3)

        return pred_pixels

    def get_query_features(
        self, dino_features, input_cameras, query_cameras, image_size,
        decoder_return_type, return_grid_rays=False, return_slt=False
    ):

        device = dino_features.device
        grid_rays, _ = get_grid_rays_gpu(
            query_cameras, image_size=image_size, min_x=1, max_x=-1,
            min_y=1, max_y=-1
        ) # grid_rays: (n_query_cameras, H*W, 6), and, xys: (n_query_cameras, H*W, 2)

        # Break the given number of query rays into reasonable batches (to avoid OOM)
        n_queries = self.num_pixel_queries
        num_query_batches = math.ceil(grid_rays.shape[1] / n_queries)
        pred_pixels_list = []

        scene_encoding = self.encode(device, dino_features, input_cameras)

        for i in range(num_query_batches):
            # Get grid rays corresponding to the current batch of pixels
            grid_rays_current_iter = grid_rays[:, i * n_queries:(i + 1) * n_queries]  # (n_query_cameras, n_queries, 6)

            # Predict the pixel values for the given rays
            # NOTE: Removed input_indices requirement
            pred_pixels = self.decode(
                scene_encoding=scene_encoding, query_pixel_rays=grid_rays_current_iter,
                input_cameras=input_cameras, device=device, return_type=decoder_return_type
            )  # (n_query_cameras, n_queries, F)
            pred_pixels_list.append(pred_pixels)

        query_features = torch.cat(pred_pixels_list, dim=1)  # (n_query_cameras, H*W, F)
        feature_dim = query_features.shape[-1]
        query_features = torch.reshape(query_features, (-1, *image_size, feature_dim))

        if return_grid_rays:
            plucker_grid_rays = get_plucker_parameterization(
                torch.reshape(grid_rays, (-1, *image_size, 6)).to(dino_features.device)
            )

        else:
            plucker_grid_rays = None

        if return_slt:
            output = (query_features, plucker_grid_rays, scene_encoding)

        else:
            output = (query_features, plucker_grid_rays)

        return output

    def convert_to_view_space(self, input_cameras, input_rays, query_rays):
        if not self.view_space:
            return input_rays, query_rays

        reference_cameras = [cameras[0] for cameras in input_cameras]
        reference_R = [camera.R.to(input_rays.device) for camera in reference_cameras] # List (length=batch_size) of Rs(shape: 1, 3, 3)
        reference_R = torch.cat(reference_R, dim=0) # (B, 3, 3)
        reference_T = [camera.T.to(input_rays.device) for camera in reference_cameras] # List (length=batch_size) of Ts(shape: 1, 3)
        reference_T = torch.cat(reference_T, dim=0) # (B, 3)
        input_rays = transform_rays(reference_R=reference_R, reference_T=reference_T, rays=input_rays)
        query_rays = transform_rays(reference_R=reference_R, reference_T=reference_T, rays=query_rays.unsqueeze(1)).squeeze(1)
        return input_rays, query_rays

    def convert_query_rays_to_view_space(self, device, input_cameras, query_rays):
        if not self.view_space:
            return query_rays

        reference_cameras = [cameras[0] for cameras in input_cameras]
        reference_R = [camera.R.to(device) for camera in reference_cameras] # List (length=batch_size) of Rs(shape: 1, 3, 3)
        reference_R = torch.cat(reference_R, dim=0) # (B, 3, 3)
        reference_T = [camera.T.to(device) for camera in reference_cameras] # List (length=batch_size) of Ts(shape: 1, 3)
        reference_T = torch.cat(reference_T, dim=0) # (B, 3)
        query_rays = transform_rays(reference_R=reference_R, reference_T=reference_T, rays=query_rays.unsqueeze(1)).squeeze(1)
        return query_rays

    def create_cameras_with_identity_extrinsics(self, input_cameras):

        identity_cameras = []
        for cameras in input_cameras:
            cur_list = []
            for camera in cameras:
                new_camera = PerspectiveCameras(
                    R=torch.eye(3, dtype=torch.float32).unsqueeze(0),
                    T=torch.zeros((1, 3), dtype=torch.float32),
                    focal_length=camera.focal_length, principal_point=camera.principal_point,
                    image_size=camera.image_size
                )
                cur_list.append(new_camera)
            identity_cameras.append(cur_list)

        return identity_cameras
    