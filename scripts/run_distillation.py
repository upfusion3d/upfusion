import os
import cv2
import torch
import imageio
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.transforms import Transform3d
from transformers import logging as transformers_logging
from pytorch3d.renderer.cameras import look_at_view_transform

from external.nerf.network_grid import NeRFNetwork
from external.sparsefusion_utils.common_utils import get_lpips_fn, normalize, unnormalize
from external.sparsefusion_utils.render_utils import init_ray_sampler
from external.sparsefusion_utils.external_utils import PerceptualLoss

from upsrt.model.model import UpSRT
from dino.model.model import DINOv2KeyExtractor
from diffusion.pipeline_control_net import DiffusionPipelineCN


#################################################################################
# Util Classes
#################################################################################

class ITWDataset(Dataset):

    def __init__(self, root):
        super().__init__()
        self.root = root
        image_size = [256, 256]
        focal_length = [4.5, 4.5]
        principal_point = [0.0, 0.0]

        files = sorted(os.listdir(self.root))
        self.masks = []
        self.images = []
        for file in files:
            img_path = os.path.join(self.root, file)
            img = torch.tensor(self.load_image(img_path).astype(np.float32)/255.0)
            img = torch.permute(img, (2, 0, 1)).contiguous()
            img = img * 2.0 - 1.0  # (3, 256, 256)

            self.images.append(img)

        self.cameras = self.create_synth_cameras(
            num_cameras=250, focal_length=focal_length,
            principal_point=principal_point, image_size=image_size, inference=False
        )
        self.inference_cameras = self.create_synth_cameras(
            num_cameras=32, focal_length=focal_length,
            principal_point=principal_point, image_size=image_size, inference=True
        )
        self.images = torch.stack(self.images, dim=0)  # (N, 3, 256, 256)
        self.input_cameras = self.create_cameras_with_identity_extrinsics(
            num_cameras = len(self.images), focal_length = focal_length,
            principal_point = principal_point, image_size = image_size,
        )

    @staticmethod
    def load_image(path):
        x = cv2.imread(path, 1)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        return x

    @staticmethod
    def create_synth_cameras(
        num_cameras, focal_length, principal_point,
        image_size, sample_distance=False, inference=False
    ):

        base_elevation = 22.5
        principal_point_ = torch.tensor([principal_point])
        focal_length_ = torch.tensor([focal_length])
        image_size_ = torch.tensor([image_size])

        synth_cameras, w1_ci_TFs = [], []
        distance_choices = [1.5, 1.0, 0.5]
        min_ele, max_ele = base_elevation - 30.0, base_elevation + 30.0

        if inference:
            azimuths_ = torch.linspace(0.0, 360.0, num_cameras+1)
            elevations_ = torch.ones((num_cameras+1,)) * base_elevation
            azimuths_, elevations_ = azimuths_[:-1], elevations_[:-1]
        else:
            azimuths_, elevations_ = None, None

        # Setting up W1 to Ci transforms
        for i in range(num_cameras):

            if inference:
                distance_choice = 1.0
                azimuth_choice = azimuths_[i]
                elevation_choice = elevations_[i]

            else:
                if i == 0:
                    azimuth_choice = 0.0
                    elevation_choice = base_elevation
                    distance_choice = 1.0
                else:
                    azimuth_choice = np.random.random() * 360.0
                    elevation_choice = np.random.random() * (max_ele - min_ele) + min_ele

                    if sample_distance:
                        distance_choice = distance_choices[np.random.randint(0, 3)]
                    else:
                        distance_choice = 1.0

            R, T = look_at_view_transform(
                dist = distance_choice, elev = elevation_choice,
                azim = azimuth_choice, degrees = True
            )

            w1_ci_TF_ = torch.eye(4).unsqueeze(0)
            w1_ci_TF_[:, :3, :3] = R
            w1_ci_TF_[:,  3, :3] = T
            w1_ci_TF = Transform3d(matrix = w1_ci_TF_)
            w1_ci_TFs.append(w1_ci_TF)

        # Location of camera corresponding to 1st image in the W2 system
        w2_c1_TF_ = torch.eye(4).unsqueeze(0)
        w2_c1_TF_[0, 3, :3] = torch.tensor([0.0, 0.0, 1.0])
        w2_c1_TF = Transform3d(matrix = w2_c1_TF_)

        # Location of camera corresponding to 1st image in the W1 system (how? because we defined it to be so!)
        w1_c1_TF = w1_ci_TFs[0]

        # Calculating W2 to W1 transform
        w2_w1_TF = w2_c1_TF.compose(w1_c1_TF.inverse())

        # Re-calculating cameras such that every camera uses W2 to Ci transform
        for i in range(num_cameras):
            w1_ci_TF = w1_ci_TFs[i]
            w2_ci_TF = w2_w1_TF.compose(w1_ci_TF)

            w2_ci_TF_ = w2_ci_TF.get_matrix()
            new_R = w2_ci_TF_[:, :3, :3]  # (1, 3, 3)
            new_T = w2_ci_TF_[:,  3, :3]  # (1, 3)

            camera = PerspectiveCameras(
                R=new_R, T=new_T, focal_length=focal_length_,
                principal_point=principal_point_, image_size=image_size_
            )
            synth_cameras.append(camera)

        return synth_cameras

    @staticmethod
    def create_cameras_with_identity_extrinsics(num_cameras, focal_length, principal_point, image_size):

        cameras = []
        principal_point_ = torch.tensor([principal_point])
        focal_length_ = torch.tensor([focal_length])
        image_size_ = torch.tensor([image_size])

        for i in range(num_cameras):
            camera = PerspectiveCameras(
                R=torch.eye(3, dtype=torch.float32).unsqueeze(0),
                T=torch.zeros((1, 3), dtype=torch.float32),
                focal_length=focal_length_, principal_point=principal_point_,
                image_size=image_size_
            )
            cameras.append(camera)

        return cameras

    def __getitem__(self, idx):
        return (self.images, self.cameras, self.input_cameras, self.inference_cameras)


class CachedQueryDataset(torch.utils.data.Dataset):
    def __init__(self, query_cameras, cache):
        self.query_cameras = query_cameras[0]
        self.cache = cache

    def __len__(self):
        return len(self.query_cameras)

    @staticmethod
    def collate_fn(batch):
        batched_cameras = concat_cameras([x["query_cameras"] for x in batch])

        return_dict = {
            "slt": torch.cat([x["slt"] for x in batch], dim = 0),
            "cond_images": torch.cat([x["cond_images"] for x in batch], dim = 0),
            "query_rgb_256": torch.cat([x["query_rgb_256"] for x in batch], dim = 0),
            "query_cameras": batched_cameras,
        }
        return return_dict

    def __getitem__(self, idx):

        return_dict = {
            "slt": self.cache[idx]["slt"],
            "cond_images": self.cache[idx]["cond_images"],
            "query_rgb_256": self.cache[idx]["query_rgb_256"],
            "query_cameras": self.query_cameras[idx]
        }
        return return_dict

#################################################################################
# Util Functions
#################################################################################

def get_cfg(cfg_path, verbose=False):
    cfg = OmegaConf.load(cfg_path)
    if verbose:
        print(OmegaConf.to_yaml(cfg))
    return cfg

def save_image(path, tensor, unnorm = False):

    img = np.transpose(tensor.cpu().numpy(), (1, 2, 0))
    if unnorm:
        img = img * 0.5 + 0.5

    img = np.clip(img, 0.0, 1.0)
    Image.fromarray((img*255.0).astype(np.uint8)).save(path)

def _collect_attr(cams,  attr):
    return torch.cat([getattr(x, attr) for x in cams], dim = 0)

def concat_cameras(list_of_cameras):
    concat_cameras = PerspectiveCameras(
        R=_collect_attr(list_of_cameras, "R"), T=_collect_attr(list_of_cameras, "T"),
        focal_length=_collect_attr(list_of_cameras, "focal_length"),
        principal_point=_collect_attr(list_of_cameras, "principal_point"),
        image_size=_collect_attr(list_of_cameras, "image_size"),
    )
    return concat_cameras

def batched_cameras_to_list(batched_cameras):
    cameras = []
    for i in range(len(batched_cameras)):
        cam = PerspectiveCameras(
            R=batched_cameras.R[i:i+1], T=batched_cameras.T[i:i+1],
            focal_length=batched_cameras.focal_length[i:i+1],
            principal_point=batched_cameras.principal_point[i:i+1],
            image_size=batched_cameras.image_size[i:i+1],
        )
        cameras.append(cam)

    return cameras

def _prepare_condition(srt_model, dino_model, input_views, input_cameras, query_cameras):

    dino_features = dino_model(input_views)
    query_features, plucker_encoding, slt = srt_model.get_query_features(
        dino_features=dino_features, input_cameras=input_cameras,
        query_cameras=query_cameras, image_size=(32, 32),
        decoder_return_type = "pre_rgb_mlp_features",
        return_grid_rays = True, return_slt = True
    )
    cond_images = torch.cat((query_features, plucker_encoding), dim = 3)
    return (cond_images.detach().cpu(), slt.detach().cpu()), dino_features

def get_default_torch_ngp_opt():
    '''
    Return default options for torch-ngp
    '''
    opt = argparse.Namespace()
    opt.cuda_ray = False
    opt.bg_radius = 0
    opt.density_thresh = 10
    opt.bound = 1
    opt.min_near = 0.05
    return opt

#################################################################################
# Main Functions
#################################################################################

def distillation_loop(
    gpu, cfg, args, opt, model_tuple,
    save_dir, seq_name, max_itr=3000, loss_fn_vgg=None,
):

    #################################################################################
    # Setup
    #################################################################################

    print("[***] Preparing training setup")
    lambda_opacity = 1e-3
    lambda_entropy = 0.0
    lambda_percep = 0.1

    plot_log_freq = 20
    img_log_freq = 100
    hw_scale = 2.0

    gradient_accumulation_steps = 1
    sds_bootstrap_itrs = 300
    start_percep_step = -1
    bg_color_choice = 1.0
    max_itr = 3000

    perceptual_loss = PerceptualLoss('vgg', device=f'cuda:{gpu}')
    fusion_loss_list = []

    # Creating directories and fetching models
    os.makedirs(os.path.join(args.out_dir, "render_imgs", seq_name), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "render_gifs"), exist_ok=True)
    srt_model, dino_model, diffusion_pipeline = model_tuple

    dataset = ITWDataset(root=args.in_dir)
    custom_imgs, custom_cameras, input_cameras, inference_cameras = dataset[0]

    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    diffusion_pipeline.handle, srt_model, dino_model = accelerator.prepare(
        diffusion_pipeline.handle, srt_model, dino_model
    )

    # Setting up data
    input_views = custom_imgs[None].to(gpu)  # (B, num_input_views, C, H, W)
    input_cameras = [[x.to(f"cuda:{gpu}") for x in input_cameras]]
    custom_cameras = [x.to(f"cuda:{gpu}") for x in custom_cameras]
    inference_cameras = [x.to(f"cuda:{gpu}") for x in inference_cameras]

    #################################################################################
    # Creating renderers
    #################################################################################
    batched_custom_cameras = concat_cameras(custom_cameras)
    cam_dist_max = torch.max(torch.linalg.norm(batched_custom_cameras.get_camera_center(), axis=1))
    min_depth = cam_dist_max * 0.1
    volume_extent_world = cam_dist_max * 1.8

    sampler_grid, _, sampler_feat = init_ray_sampler(
        gpu, 256, 256, min=min_depth,
        max=volume_extent_world, scale_factor=hw_scale
    )

    #################################################################################
    # Caching condition inputs
    #################################################################################
    cache = dict()
    num_cameras = len(custom_cameras)
    for q_idx in tqdm(range(num_cameras), desc="Caching condition inputs"):
        with torch.no_grad():
            condition_, dino_features_ = _prepare_condition(
                srt_model, dino_model, input_views=input_views[:1],
                input_cameras=input_cameras[:1],
                query_cameras=[custom_cameras[q_idx]]
            )

            # Currently doing two forward passes! Inefficient but it is easier to code for now.
            query_rgb_256, _ = srt_model.get_query_features(
                dino_features=dino_features_, input_cameras=input_cameras[:1],
                query_cameras=[custom_cameras[q_idx]], image_size=(256, 256),
                decoder_return_type = "pred_rgb",
                return_grid_rays = False
            )

            cond_images_, slt_ = condition_
            cache[q_idx] = {
                "cond_images": cond_images_, "slt": slt_,
                "query_rgb_256": torch.permute(query_rgb_256, (0, 3, 1, 2)).contiguous().cpu()
            }

    #################################################################################
    # Creating data loaders
    #################################################################################

    # Creating the full query data loader
    custom_cameras_cpu = [x.cpu() for x in custom_cameras]
    query_dataset = CachedQueryDataset(query_cameras = [custom_cameras_cpu], cache = cache)
    query_loader = DataLoader(
        dataset=query_dataset, batch_size=1, shuffle=True,
        num_workers=0, pin_memory=True, collate_fn=CachedQueryDataset.collate_fn
    )

    # Preparing the data loaders
    query_loader = accelerator.prepare(query_loader)
    query_itr = iter(query_loader)

    # Setup the NGP model
    ngp_network = NeRFNetwork(opt).cuda(gpu).train()
    optimizer = torch.optim.Adam(ngp_network.get_params(lr=5e-4))
    optimizer = accelerator.prepare(optimizer)
    print("[***] Training setup prepared!")

    #################################################################################
    # Training loop
    #################################################################################

    for itr in tqdm(range(max_itr), desc="Training"):

        if itr < start_percep_step:
            lambda_percep_ = 0.0
        elif itr >= start_percep_step:
            lambda_percep_ = lambda_percep

        with accelerator.accumulate(ngp_network):

            total_loss, sds_loss = 0.0, 0.0
            ngp_network.train()
            if opt.cuda_ray and itr % 16 == 0:
                ngp_network.update_extra_state()

            # Fetching a batch of data from the query data loader
            try:
                batch = next(query_itr)
            except StopIteration:
                query_itr = iter(query_loader)
                batch = next(query_itr)

            #################################################################
            # Computing SDS loss
            #################################################################

            batch_slt = batch["slt"]
            batch_cameras = batch["query_cameras"]
            batch_cond_images = batch["cond_images"]

            # Fetch rays
            ray_bundle = sampler_feat(batch_cameras)
            H, W = ray_bundle.origins.shape[1], ray_bundle.origins.shape[2]
            rays_o = rearrange(ray_bundle.origins, 'b h w c -> b (h w) c')  # [B, N, 3]
            rays_d = rearrange(ray_bundle.directions, 'b h w c -> b (h w) c')  # [B, N, 3]
            B, N = rays_o.shape[:2]

            # Render image
            bg_color = bg_color_choice
            outputs = ngp_network.render(
                rays_o, rays_d, staged=False, perturb=True, bg_color=bg_color,
                ambient_ratio=1.0, shading='albedo', force_all_rays=True, **vars(opt)
            )
            rendered_images = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()  # [1, 3, H, W]
            rendered_silhouettes = outputs['weights_sum'].reshape(B, H, W, 1).permute(0, 3, 1, 2).contiguous()

            rendered_images = F.interpolate(rendered_images, scale_factor=hw_scale, mode='bilinear')
            rendered_silhouettes = F.interpolate(rendered_silhouettes, scale_factor=hw_scale, mode='bilinear')

            # Compute SDS loss
            ddim_steps = 30
            t_end = ddim_steps - 1
            t_start = ddim_steps - 5 if itr < sds_bootstrap_itrs else 1 # Bootstrapping with high noise

            with torch.no_grad():
                srt_cond = (batch_cond_images, batch_slt, [batch_cameras]) # DF+SLT Conditioning!
                normed_pred, alpha_cumprod = diffusion_pipeline.forward_multi_step_denoise(
                    clean_data=normalize(rendered_images), srt_cond=srt_cond, batch_size=1,
                    unconditional_guidance_scale=cfg.diffusion.model.unconditional_guidance_scale,
                    ddim_steps=ddim_steps, t_start=t_start, t_end=t_end
                )
                pred_img = unnormalize(normed_pred).clip(0.0, 1.0)

            fusion_weight = (1 - alpha_cumprod).to(pred_img.device)

            if itr < sds_bootstrap_itrs:
                fusion_weight = 1.0  # Use fusion_weight 1.0 when bootstrapping with high noise

            sds_loss = fusion_weight * ((rendered_images - pred_img)**2).mean()

            if lambda_percep_ > 0.0:
                percep_term = perceptual_loss(rendered_images, pred_img, normalize=True)
                sds_loss += percep_term.mean() * lambda_percep_

            fusion_loss_list.append(sds_loss.item())

            # Compute regularizing losses
            # Regularizing loss 1: Opacity loss
            opacity_term = torch.zeros((1)).cuda(gpu).requires_grad_()
            if lambda_opacity > 0:
                opacity_term = torch.sqrt((rendered_silhouettes ** 2) + .01).mean()

            # Regularizing loss 1: Entropy loss
            entropy_term = torch.zeros((1)).cuda(gpu).requires_grad_()
            if lambda_entropy > 0:
                alphas = (rendered_silhouettes).clamp(1e-5, 1 - 1e-5)
                entropy_term = (- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).mean()

            total_loss = sds_loss + lambda_opacity * opacity_term + lambda_entropy * entropy_term

            accelerator.backward(total_loss)
            optimizer.step()
            optimizer.zero_grad()

        # Log plots
        if itr % plot_log_freq == 0:
            plt.plot(list(range(len(fusion_loss_list))), fusion_loss_list, linewidth=1)
            save_path = os.path.join(args.out_dir, "log", f"{seq_name}_fusionloss.jpg")
            plt.savefig(save_path)
            plt.cla()
            plt.close()

        # Log images
        if itr % img_log_freq == 0:

            with torch.no_grad():

                # Fetch rays
                ngp_network.eval()
                ray_bundle = sampler_grid(batch_cameras)
                H, W = ray_bundle.origins.shape[1], ray_bundle.origins.shape[2]
                rays_o = rearrange(ray_bundle.origins, 'b h w c -> b (h w) c')  # [B, N, 3]
                rays_d = rearrange(ray_bundle.directions, 'b h w c -> b (h w) c')  # [B, N, 3]
                B, N = rays_o.shape[:2]

                # Render image
                bg_color = bg_color_choice
                outputs = ngp_network.render_batched(
                    rays_o, rays_d, batched=True, perturb=True, bg_color=bg_color,
                    ambient_ratio=1.0, shading='albedo', force_all_rays=True, **vars(opt)
                )
                rendered_images = outputs['image'].reshape(B, H, W, 3).contiguous()[0] # [1, 3, H, W]
                rendered_silhouettes = outputs['weights_sum'].reshape(B, H, W, 1).contiguous()[0]

            rendered_image_vis = rendered_images.detach().cpu().numpy()
            rendered_sil_vis = rendered_silhouettes.expand(-1, -1, 3).detach().cpu().numpy()
            render_vis = np.hstack((rendered_image_vis, rendered_sil_vis))

            save_path = os.path.join(args.out_dir, "log", f"{seq_name}_vis_{str(itr).zfill(5)}.jpg")
            imageio.imwrite(save_path, (render_vis*255).astype(np.uint8))

    #################################################################################
    # Post Training Visualization
    #################################################################################

    circle_rgb_list, circle_sil_list = [], []
    for ci in tqdm(range(len(inference_cameras)), desc="Rendering"):

        render_camera = inference_cameras[ci].to(f"cuda:{gpu}")
        with torch.no_grad():

            # Fetch rays
            ngp_network.eval()
            ray_bundle = sampler_grid(render_camera)
            H, W = ray_bundle.origins.shape[1], ray_bundle.origins.shape[2]
            rays_o = rearrange(ray_bundle.origins, 'b h w c -> b (h w) c')  # [B, N, 3]
            rays_d = rearrange(ray_bundle.directions, 'b h w c -> b (h w) c')  # [B, N, 3]
            B, N = rays_o.shape[:2]

            # Render image
            bg_color = bg_color_choice
            outputs = ngp_network.render_batched(
                rays_o, rays_d, batched=True, perturb=True, bg_color=bg_color,
                ambient_ratio=1.0, shading='albedo', force_all_rays=True, **vars(opt)
            )
            rendered_images = outputs['image'].reshape(B, H, W, 3) # [1, H, W, 3]
            rendered_silhouettes = outputs['weights_sum'].reshape(B, H, W, 1) # [1, H, W, 3]
            rendered_images = rendered_images[0].detach().cpu().numpy()
            rendered_silhouettes = rendered_silhouettes.expand(-1, -1, -1, 3)[0].detach().cpu().numpy()

        circle_rgb_list.append((rendered_images * 255.0).astype(np.uint8))
        circle_sil_list.append((rendered_silhouettes * 255.0).astype(np.uint8))

    fps = 10.0
    rgb_gif_path = os.path.join(save_dir, "render_gifs", f"{seq_name}_rgb.gif")
    sil_gif_path = os.path.join(save_dir, "render_gifs", f"{seq_name}_sil.gif")
    imageio.mimwrite(rgb_gif_path, circle_rgb_list, duration = 1000 * (1/fps), loop = 0)
    imageio.mimwrite(sil_gif_path, circle_sil_list, duration = 1000 * (1/fps), loop = 0)

    #################################################################################
    # Saving torch NGP model
    #################################################################################
    model_path = os.path.join(save_dir, f"{seq_name}.pt")
    torch.save({'model_state_dict': ngp_network.state_dict()}, model_path)


def run(args):

    gpu = 0
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'log'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'render_imgs'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'render_gifs'), exist_ok=True)

    seq_name = args.name
    if args.config_path is None:
        raise ValueError("Please provide path to a config file.")

    cfg = get_cfg(args.config_path)
    opt = get_default_torch_ngp_opt()
    loss_fn_vgg = get_lpips_fn()

    # Setup models
    print("[***] Setting up models")
    srt_model = UpSRT(cfg.srt.model)
    dino_model = DINOv2KeyExtractor(cfg.dino.model)
    diffusion_pipeline = DiffusionPipelineCN(
        cfg.diffusion.model, srt_model=srt_model,
        dino_model=dino_model
    )
    srt_model.eval()
    dino_model.eval()
    diffusion_pipeline.handle.eval()
    model_tuple = (srt_model, dino_model, diffusion_pipeline)

    # Loading model weights
    upsrt_load_dict = torch.load(os.path.join(args.weights_dir, "upsrt.pt"), map_location="cpu")
    srt_model.load_state_dict(upsrt_load_dict['model_state_dict'])

    diffusion_load_dict = torch.load(os.path.join(args.weights_dir, "upfusion2d.pt"), map_location="cpu")
    diffusion_pipeline.handle.load_state_dict(diffusion_load_dict['model_state_dict'])
    print("[***] Model setup is complete!")

    distillation_loop(
        gpu = gpu, cfg = cfg, args = args, opt = opt, model_tuple = model_tuple, save_dir = args.out_dir,
        seq_name = seq_name, loss_fn_vgg=loss_fn_vgg
    )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--in_dir', type=str,
        required=True, help='Path to the directory with masked sparse-view images of the object.'
    )
    parser.add_argument(
        '--out_dir', type=str,
        required=True, help='Path to the directory where outputs should be stored.'
    )
    parser.add_argument(
        '--name', type=str,
        required=True, help='Name of the experiment.'
    )
    parser.add_argument(
        '--weights_dir', type=str, required=True,
        help='Path to the directory with the model weights.'
    )
    parser.add_argument(
        '--config-path', type=str,
        default="./configs/inference.yaml", help='Path to a config file.'
    )
    args = parser.parse_args()
    run(args=args)

if __name__ == "__main__":

    transformers_logging.set_verbosity_error()
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
    main()
