import torch
import torch.nn as nn

from einops import rearrange
from torchvision import transforms
from pytorch3d.renderer.cameras import PerspectiveCameras

from dino.renderer.rays import get_patch_rays, positional_encoding
from dino.utils.id_encoding import create_patch_id_encoding, create_camera_id_encoding

class DINOv2KeyExtractor(nn.Module):

    def __init__(self, cfg):

        super().__init__()
        self.cache = None
        self.model_key = cfg.model_key
        self.layer_name = cfg.layer_name

        self.model = torch.hub.load('facebookresearch/dinov2', self.model_key)
        self.model.eval()

        self.modules_dict = dict([*self.model.named_modules()])
        mod = self.modules_dict[self.layer_name]
        mod.register_forward_hook(self.create_hook(self.layer_name))

        self.dino_transforms = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def create_hook(self, layer_name):
        """
        Creates a hook function for the given layer name that saves the output of
        that layer to self.cache
        """
        def hook_fn(module, inp, out):

            # Shape details
            _, _, dim3 = out.shape
            dim = int(dim3 / 3)

            # Removing the CLS token and extracting only the key features
            # and storing it in self.cache
            self.cache = out[:, 1:, dim:(dim+dim)]

        return hook_fn

    def preprocess(self, x):
        """
        Args:
            x   :   torch.Tensor with shape [B, C, H, W] and values in the range [-1, 1]
        """
        unnorm_img = x * 0.5 + 0.5 # unnorm is in the range [0, 1]
        output = self.dino_transforms(unnorm_img)
        return output

    def get_key_features(self, x):
        """
        Args:
            x   :   torch.Tensor with shape [B, C, H, W] and values in the range [-1, 1]
        """
        _ = self.model(self.preprocess(x))
        return self.cache # (B, T, D)

    def forward(self, input_views):
        """
        Args:
            input_views :   torch.Tensor with shape [B, N, C, H, W] and values in the range [-1, 1]
        """
        N = input_views.shape[1]
        reshaped = rearrange(input_views, "b n c h w -> (b n) c h w")

        key_feats_ = self.get_key_features(reshaped) # (B*N, T, K)
        key_feats = rearrange(key_feats_, "(b n) t d -> b n t d", n=N)

        return key_feats


class DLTExtractor(nn.Module):

    # Reference: https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904
    def __init__(self, cfg):

        super().__init__()

        self.model_key = cfg.model_key
        self.num_patches_x = cfg.num_patches_x
        self.num_patches_y = cfg.num_patches_y
        self.num_freqs = cfg.ray.num_freqs
        self.start_freq = cfg.ray.start_freq
        self.parameterize = cfg.ray.parameterize
        self.harmonic_embedding_dim = 2 * self.num_freqs * 6

        if self.model_key != "dinov2_vitb14":
            raise ValueError

        key_features_dims = 768 # This is specific to dinov2_vitb14
        self.linear_scene = nn.Linear(
            key_features_dims + self.harmonic_embedding_dim + 2*self.num_freqs + 2*self.num_freqs,
            cfg.out_dim
        )

    def forward(self, device, dino_features, input_cameras):

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

        # Concatenate encodings
        dlt_ = torch.cat(
            (dino_features, input_patch_rays, patch_id_encoding, camera_id_encoding), dim=-1
        ) # (b, n_inp, patch, K + pos_embedding_dim + 2*self.num_freqs + 2*self.num_freqs)
        dlt_ = self.linear_scene(dlt_)

        dlt = rearrange(dlt_, "b n p d -> b (n p) d")

        return dlt

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
