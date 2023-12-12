import torch
import torch.nn as nn
from control_net.cldm.model import create_model, load_state_dict
from control_net.ldm.models.diffusion.ddim import DDIMSampler

class DiffusionPipelineCN(nn.Module):

    def __init__(self, cfg, srt_model=None, dino_model=None):

        super().__init__()
        self.cfg = cfg
        self.control_net_model_config_path = self.cfg.control_net_model_config_path
        self.prompt_color = self.cfg.control_net_prompt_color
        self._setup_model()
        self.srt_model = srt_model
        self.dino_model = dino_model

        self.cond_type = self.cfg.cond_type
        if self.cond_type == "DF":
            self._create_batch_dict_fn = self._create_batch_dict_df
            self._maybe_dropout_condition_fn = self._maybe_dropout_condition_df

        elif self.cond_type == "SLT":
            self._create_batch_dict_fn = self._create_batch_dict_slt
            self._maybe_dropout_condition_fn = self._maybe_dropout_condition_slt

        elif self.cond_type == "DF+SLT":
            self._create_batch_dict_fn = self._create_batch_dict_dfslt
            self._maybe_dropout_condition_fn = self._maybe_dropout_condition_dfslt

        else:
            raise ValueError

    def _setup_model(self):

        model = create_model(self.cfg.control_net_model_config_path).cpu()
        model.sd_locked = self.cfg.control_net_sd_locked
        model.only_mid_control = False

        # if self.cfg.control_net_init_ckpt_path is not None:
        #     model.load_state_dict(load_state_dict(self.cfg.control_net_init_ckpt_path, location='cpu'))
        #     model.sd_locked = self.cfg.control_net_sd_locked
        #     model.only_mid_control = False
        # else:
        #     raise RuntimeError

        self.handle = model

    def to_device(self, device):
        self.dino_model = self.dino_model.to(device)
        self.srt_model = self.srt_model.to(device)
        self.handle = self.handle.to(device)

    def _get_text_prompt(self, batch_size, class_idxs=None):
        prompt = [f"a high quality image with a {self.prompt_color} background" for _ in range(batch_size)]

        return prompt

    def _get_null_text_prompt(self, batch_size):
        prompt = ["" for _ in range(batch_size)]
        return prompt

    def _create_batch_dict_df(self, clean_data, srt_cond, class_idxs=None):
        # NOTE: clean_data and cond_images must be channels last!
        batch = {
            "jpg": clean_data,
            "txt": self._get_text_prompt(len(clean_data), class_idxs),
            "hint": srt_cond, # srt_cond is cond_images
        }
        return batch

    def _create_batch_dict_slt(self, clean_data, srt_cond, class_idxs=None):
        # NOTE: clean_data must be channels last!
        slt, query_cameras = srt_cond
        batch = {
            "jpg": clean_data,
            "txt": self._get_text_prompt(len(clean_data), class_idxs),
            "slt": slt,
            "query_cameras": query_cameras,
        }
        return batch

    def _create_batch_dict_dfslt(self, clean_data, srt_cond, class_idxs=None):
        # NOTE: clean_data must be channels last!
        cond_images, slt, query_cameras = srt_cond
        batch = {
            "jpg": clean_data,
            "txt": self._get_text_prompt(len(clean_data), class_idxs),
            "hint": cond_images,
            "slt": slt,
            "query_cameras": query_cameras,
        }
        return batch

    def _maybe_dropout_condition_df(self, batch, cfg_seed, condition_dropout):

        # Logic inspired from https://github.com/cvlab-columbia/zero123/blob/main/zero123/ldm/models/diffusion/ddpm.py
        prompt, cond_images = batch["txt"], batch["hint"]
        random_number = torch.rand((), generator = torch.Generator().manual_seed(cfg_seed)).item()
        drop_prompt = random_number < (2 * condition_dropout)
        drop_condition = (random_number >= condition_dropout) and (random_number < (3 * condition_dropout))

        if drop_prompt:
            prompt = self._get_null_text_prompt(len(prompt))

        if drop_condition:
            cond_images = torch.zeros_like(cond_images)

        return batch

    def _maybe_dropout_condition_slt(self, batch, cfg_seed, condition_dropout):

        # Logic inspired from https://github.com/cvlab-columbia/zero123/blob/main/zero123/ldm/models/diffusion/ddpm.py
        prompt, slt, query_cameras = batch["txt"], batch["slt"], batch["query_cameras"]
        random_number = torch.rand((), generator = torch.Generator().manual_seed(cfg_seed)).item()
        drop_prompt = random_number < (2 * condition_dropout)
        drop_condition = (random_number >= condition_dropout) and (random_number < (3 * condition_dropout))

        if drop_prompt:
            batch["txt"] = self._get_null_text_prompt(len(prompt))

        if drop_condition:
            batch["slt"] = torch.zeros_like(slt)
            batch["query_cameras"] = [None for _ in range(len(query_cameras))]

        return batch

    def _maybe_dropout_condition_dfslt(self, batch, cfg_seed, condition_dropout):

        # Logic inspired from https://github.com/cvlab-columbia/zero123/blob/main/zero123/ldm/models/diffusion/ddpm.py
        prompt, slt, query_cameras, cond_images = batch["txt"], batch["slt"], batch["query_cameras"], batch["hint"]
        random_number = torch.rand((), generator = torch.Generator().manual_seed(cfg_seed)).item()
        drop_prompt = random_number < (2 * condition_dropout)
        drop_condition = (random_number >= condition_dropout) and (random_number < (3 * condition_dropout))

        if drop_prompt:
            batch["txt"] = self._get_null_text_prompt(len(prompt))

        if drop_condition:
            batch["hint"] = torch.zeros_like(cond_images)
            batch["slt"] = torch.zeros_like(slt)
            batch["query_cameras"] = [None for _ in range(len(query_cameras))]

        return batch

    def forward_with_loss(
        self, clean_data, srt_cond, class_idxs=None,
        enable_cfg=False, cfg_seed=None, condition_dropout=0.0
    ):
        clean_data_ = torch.permute(clean_data, (0, 2, 3, 1)).contiguous()
        batch = self._create_batch_dict_fn(clean_data_, srt_cond, class_idxs)

        if enable_cfg:
            batch = self._maybe_dropout_condition_fn(batch, cfg_seed, condition_dropout)

        # This should call the shared_step function implemented in the class LatentDiffusion via inheritance
        loss, _ = self.handle.shared_step(batch)
        return loss

    def forward_one_step_denoise(self, clean_data, cond_images, class_idxs=None):

        # NOTE: Does not perform CFG!
        clean_data_ = torch.permute(clean_data, (0, 2, 3, 1)).contiguous()
        batch = self._create_batch_dict(clean_data_, cond_images, class_idxs)

        # This should call the perform_one_step_denoise function implemented in the class LatentDiffusion via inheritance
        pred_latent, t = self.handle.perform_one_step_denoise(batch)
        decoded = self.handle.decode_first_stage(pred_latent)
        alpha_cumprod = self.handle.alphas_cumprod[t]
        return decoded, alpha_cumprod

    def _prepare_srt_cond_dict(self, srt_cond):

        if self.cond_type == "DF":
            cond_images = torch.permute(srt_cond, (0, 3, 1, 2)).contiguous()
            srt_cond_dict = {
                "c_concat": [cond_images]
            }
        elif self.cond_type == "SLT":
            slt, query_cameras = srt_cond
            srt_cond_dict = {
                "slt": slt,
                "query_cameras": query_cameras
            }
        elif self.cond_type == "DF+SLT":
            cond_images, slt, query_cameras = srt_cond
            cond_images = torch.permute(cond_images, (0, 3, 1, 2)).contiguous()
            srt_cond_dict = {
                "hint": cond_images,
                "slt": slt,
                "query_cameras": query_cameras
            }
        else:
            raise ValueError

        return srt_cond_dict

    def _prepare_srt_uncond_dict(self, srt_cond):

        if self.cond_type == "DF":
            cond_images = torch.permute(srt_cond, (0, 3, 1, 2)).contiguous()
            srt_uncond_dict = {
                "c_concat": [torch.zeros_like(cond_images)]
            }

        elif self.cond_type == "SLT":
            slt, query_cameras = srt_cond
            srt_uncond_dict = {
                "slt": torch.zeros_like(slt),
                "query_cameras": [None for _ in range(len(query_cameras))]
            }

        elif self.cond_type == "DF+SLT":
            cond_images, slt, query_cameras = srt_cond
            cond_images = torch.permute(cond_images, (0, 3, 1, 2)).contiguous()
            srt_uncond_dict = {
                "hint": torch.zeros_like(cond_images),
                "slt": torch.zeros_like(slt),
                "query_cameras": [None for _ in range(len(query_cameras))]
            }

        else:
            raise ValueError

        return srt_uncond_dict

    def _prepare_args_for_cfg(self, batch_size, cond_type, c_cross, srt_cond):

        # Preparing the cond variable
        if cond_type != "F1":
            raise ValueError("Not Supported.")

        uc_cross = self.handle.get_unconditional_conditioning(batch_size)
        cond = {
            "c_crossattn": [c_cross],
            **self._prepare_srt_cond_dict(srt_cond)
        }
        uncond = {
            "c_crossattn": [uc_cross],
            **self._prepare_srt_uncond_dict(srt_cond)
        }

        return cond, uncond

    def forward_multi_step_denoise(
        self, clean_data, srt_cond, batch_size,
        unconditional_guidance_scale, cfg_type="F1", class_idxs=None,
        t_start=None, t_end=None, ddim_eta=0.0, ddim_steps=30
    ):

        ddim_sampler = DDIMSampler(self.handle)
        ddim_sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

        t_start_ = 1 if t_start is None else t_start
        t_end_ = len(ddim_sampler.ddim_timesteps)-1 if t_end is None else t_end

        # Adding noise
        encoder_posterior = self.handle.encode_first_stage(clean_data)
        x = self.handle.get_first_stage_encoding(encoder_posterior).detach()
        t = torch.randint(
            t_start_, t_end_,
            (clean_data.shape[0],), device=self.handle.device
        ).long()
        noisy_data = ddim_sampler.stochastic_encode(x, t)

        text = self._get_text_prompt(batch_size, class_idxs)
        c_cross = self.handle.get_learned_conditioning(text)
        cond, uncond = self._prepare_args_for_cfg(batch_size, cfg_type, c_cross, srt_cond)

        denoised_data = ddim_sampler.decode(
            noisy_data, cond, t, unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=uncond, cfg_type=cfg_type
        )
        decoded = self.handle.decode_first_stage(denoised_data)
        alpha_cumprod = ddim_sampler.ddim_alphas[t.item()]

        return decoded, alpha_cumprod

    def infer(
        self, srt_cond, batch_size, device, cfg_type,
        unconditional_guidance_scale, class_idxs=None
    ):
        """
        v2 performs classifier free guidance
        """
        ddim_eta = 0.0
        ddim_steps = 50

        if batch_size != 1:
            raise ValueError

        if unconditional_guidance_scale is None:
            raise ValueError

        text = self._get_text_prompt(batch_size, class_idxs)
        c_cross = self.handle.get_learned_conditioning(text)
        cond, uncond = self._prepare_args_for_cfg(batch_size, cfg_type, c_cross, srt_cond)

        infered_out = self.use_ddim_sampler(
            ddim_steps=ddim_steps, batch_size=batch_size, cond=cond,
            unconditional_guidance_scale=unconditional_guidance_scale,
            eta=ddim_eta, unconditional_conditioning=uncond,
            cfg_type=cfg_type
        )
        return infered_out

    def use_ddim_sampler(self, ddim_steps, batch_size, cond, **kwargs):

        ddim_sampler = DDIMSampler(self.handle)
        shape = (4, *self.cfg.query_feature_size)
        samples, _ = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        infered_out = self.handle.decode_first_stage(samples)
        return infered_out
