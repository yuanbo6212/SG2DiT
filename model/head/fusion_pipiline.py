# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
from model.diffusers.schedulers.scheduling_ddim import DDIMScheduler
from model.diffusers.schedulers.scheduling_deis_multistep import DEISMultistepScheduler
from model.diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from model.diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from model.diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from model.diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
from model.diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from model.diffusers.schedulers.scheduling_heun_discrete import HeunDiscreteScheduler
from model.diffusers.schedulers.scheduling_pndm import PNDMScheduler

from typing import Union, Dict, Tuple, Optional
from mmengine.model import BaseModule
from model.loss import loss
from model.head.fusion_head import Fusion_head
from model.head.DFT_arch import DFT


class Fusion_Pipiline(BaseModule):
    def __init__(
            self,
            LDM_Enc,
            LDM_Dec,
            sample_selected,
            model_selected,
            feat_channels,
            inference_steps=5,  # 推理步数
            num_train_timesteps=1000,  # 训练总步数（scheduler）
            mode='Max',
            fusion_task='MEF',
            channel_emdin=8,
            num_blocks=[4, 4, 4, 4],
            heads=[1, 2, 4, 8],
            bias=False,
            LayerNorm_type='WithBias',
    ):
        super().__init__()
        self.mode = mode
        self.LDM_Enc = LDM_Enc
        self.LDM_Dec = LDM_Dec
        # 统一小写处理，方便配置中不同写法
        self.sample_selected = sample_selected
        self.sample_key = sample_selected.lower() if isinstance(sample_selected, str) else ''
        self.fusion_task = fusion_task
        self.model_selected = model_selected
        if self.fusion_task == 'MEF':
            ffn_factor = 2.66
        else:
            ffn_factor = 4.0
        # Sampling step
        self.diffusion_inference_steps = inference_steps

        # Denoising network
        if self.model_selected == 'DFT':
            self.model = DFT(inp_channels=channel_emdin * 2,
                             out_channels=channel_emdin * 2,
                             dim=feat_channels[0],
                             num_blocks=num_blocks,
                             heads=heads,
                             ffn_factor=ffn_factor,
                             bias=bias,
                             LayerNorm_type=LayerNorm_type,
                             num_channel=feat_channels)

        # Fusion head
        self.fusion = Fusion_head(
            out_channels=8,
            dim=128,
            num_blocks=[2, 2, 2, 2],
            heads=[1, 2, 4, 8],
            ffn_factor=2.66,
            bias=False,
            LayerNorm_type='WithBias',
        )

        # Diffusion scheduler/sampler  —— 兼容多种别名，不区分大小写
        key = self.sample_key
        if 'ddim' in key or key == 'ddim':
            self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False)
        elif 'dpmsolver' in key or 'ddp-solver' in key:
            # 区分单步与多步（含 ++ 表示 multistep）
            if '++' in key or 'multistep' in key:
                self.scheduler = DPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps)
            else:
                self.scheduler = DPMSolverSinglestepScheduler(num_train_timesteps=num_train_timesteps)
        elif 'deis' in key:
            self.scheduler = DEISMultistepScheduler(num_train_timesteps=num_train_timesteps)
        elif 'unipc' in key or 'unipc' in key:
            self.scheduler = UniPCMultistepScheduler(num_train_timesteps=num_train_timesteps)
        elif 'lms' in key:
            self.scheduler = LMSDiscreteScheduler(num_train_timesteps=num_train_timesteps)
        elif 'heun' in key:
            self.scheduler = HeunDiscreteScheduler(num_train_timesteps=num_train_timesteps)
        elif 'pndm' in key:
            self.scheduler = PNDMScheduler(num_train_timesteps=num_train_timesteps)
        elif 'euler' in key:
            self.scheduler = EulerDiscreteScheduler(num_train_timesteps=num_train_timesteps)
        else:
            # 默认回退为 DDIM（保证不为 None）
            self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False)

        # Iterative denoising framework
        self.pipeline = DenoisePipiline(self.model, self.scheduler, self.sample_selected)

    def set_loss(self, device):
        # Diffusion loss
        self.dif_loss = nn.MSELoss().to(device)
        # Fusion loss
        if self.fusion_task == 'MEF' or self.fusion_task == 'MFF':
            self.fusion_loss = loss.Fusion_loss(mode=self.mode, lambda1=10, lambda2=20, lambda3=20).to(device)
        else:
            self.fusion_loss = loss.Fusion_loss(mode=self.mode, lambda1=10, lambda2=40, lambda3=40).to(device)

    def _process_encoder_output(self, encoder_output):
        """处理编码器输出，兼容不同格式"""
        if len(encoder_output) == 3:
            # 原有编码器格式 (U_LDM)
            x1, x2, h = encoder_output
            prior_info = None
        elif len(encoder_output) == 4:
            # 增强编码器格式 (Enhanced_U_LDM)
            x1, x2, h, prior_info = encoder_output
        else:
            raise ValueError(f"Unexpected encoder output format: {len(encoder_output)} values")

        return x1, x2, h, prior_info

    def test_Fusion(self, x_in, device):
        x_vis = x_in[:, :1]
        x_ir = x_in[:, 1:]
        with torch.no_grad():
            batch_size = x_vis.shape[0]
            device = x_vis.device
            dtype = x_vis.dtype

            # === 修改：兼容不同编码器格式 ===
            encoder_output = self.LDM_Enc(x_vis, x_ir)
            x1, x2, h, prior_info = self._process_encoder_output(encoder_output)

            latent = torch.cat((x1, x2), dim=1)
            # Denoising process
            latent_result, _, _, middle_feat = self.pipeline(
                batch_size=batch_size,
                device=device,
                dtype=dtype,
                image=latent,
                num_inference_steps=self.diffusion_inference_steps,  # 扩散步骤
                return_dict=False)
            Fusion_result = self.fusion(middle_feat, latent_result)

            # === 修改：解码器也需要处理新格式 ===
            # 检查解码器是否接受prior_info参数
            try:
                # 尝试传递prior_info（如果解码器支持）
                Fusion_result = self.LDM_Dec(Fusion_result, h, prior_info)
            except TypeError:
                # 如果解码器不支持prior_info，使用原有格式
                Fusion_result = self.LDM_Dec(Fusion_result, h)

        return Fusion_result

    def forward(self, image_vis, image_ir):
        batch_size = image_vis.shape[0]
        device = image_vis.device
        dtype = image_vis.dtype

        # === 修改：兼容不同编码器格式 ===
        encoder_output = self.LDM_Enc(image_vis, image_ir)
        x1, x2, h, prior_info = self._process_encoder_output(encoder_output)

        latent = torch.cat((x1, x2), dim=1)
        # Denoising process
        latent_result, latent_noise, pre_noise, middle_feat = self.pipeline(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            image=latent,
            num_inference_steps=self.diffusion_inference_steps,
            return_dict=False)

        # Loss calculation
        dif_loss = self.dif_loss(pre_noise, latent_noise)
        Fusion_result = self.fusion(middle_feat, latent_result)

        # === 修改：解码器也需要处理新格式 ===
        # 检查解码器是否接受prior_info参数
        try:
            # 尝试传递prior_info（如果解码器支持）
            Fusion_result = self.LDM_Dec(Fusion_result, h, prior_info)
        except TypeError:
            # 如果解码器不支持prior_info，使用原有格式
            Fusion_result = self.LDM_Dec(Fusion_result, h)

        fusion_loss, loss_gradient, loss_l1, loss_SSIM = self.fusion_loss(image_vis, image_ir, Fusion_result)

        # === 可选：添加物理先验损失 ===
        # 如果有物理先验信息，可以添加额外的损失项
        additional_loss = 0.0
        if prior_info is not None:
            # 这里可以添加基于物理先验的损失计算
            # 例如：频域损失、结构损失等
            # additional_loss = self._compute_prior_loss(prior_info, image_vis, image_ir)
            pass

        total_loss = fusion_loss + dif_loss + additional_loss

        output = {
            'Fusion': Fusion_result,
            'loss': total_loss,
            'loss_gradient': loss_gradient,
            'loss_l1': loss_l1,
            'loss_SSIM': loss_SSIM,
            'dif_loss': dif_loss,
            'prior_info': prior_info  # 可选：返回先验信息用于调试
        }

        # 如果有额外损失，也添加到输出中
        if additional_loss > 0:
            output['additional_loss'] = additional_loss

        return output

    def _compute_prior_loss(self, prior_info, image_vis, image_ir):
        """计算基于物理先验的损失（可选扩展）"""
        prior_loss = 0.0

        if prior_info is None:
            return prior_loss

        # 频域一致性损失
        if 'ir_freq_feat' in prior_info and 'vis_freq_feat' in prior_info:
            # 频域特征应该保持源图像的频域特性
            # 这里只是示例，实际损失需要根据具体需求设计
            freq_consistency_loss = torch.nn.functional.mse_loss(
                prior_info['ir_freq_feat'],
                prior_info['vis_freq_feat']
            )
            prior_loss += 0.1 * freq_consistency_loss

        # 结构一致性损失
        if 'ir_guidance_info' in prior_info and 'vis_guidance_info' in prior_info:
            # 各向异性引导信息的一致性
            # 这里只是示例
            pass

        return prior_loss


# D_t->D_t-1->D_t-2->....D_0
class DenoisePipiline:
    def __init__(self, model, scheduler, sample_selected):
        # 这里不继承 nn.Module，所以不调用 super().__init__
        self.model = model
        self.scheduler = scheduler
        self.sample_selected = sample_selected
        # 存一个方便比较的小写 key
        self.sample_key = sample_selected.lower() if isinstance(sample_selected, str) else ''

    def __call__(
            self,
            batch_size,
            device,
            dtype,
            image,
            generator: Optional[torch.Generator] = None,
            eta: float = 0.0,
            num_inference_steps: int = 50,
            return_dict: bool = True,

    ) -> Union[Dict, Tuple]:
        # generator/device 检查：使用传入的 device 而不是 self.device
        if generator is not None:
            try:
                # generator.device may not exist for CPU Generator; guard it
                gen_dev = getattr(generator, 'device', None)
                if gen_dev is not None and gen_dev.type != device.type and device.type != "mps":
                    message = (
                        f"The `generator` device is `{gen_dev}` and does not match the pipeline "
                        f"device `{device}`, so the `generator` will be ignored. "
                        f'Please use `generator=torch.Generator(device="{device}")` instead.'
                    )
                    # 不直接 raise，让上层决定是否继续；这里只发 warning 然后忽略 generator
                    # raise RuntimeError("generator.device == 'cpu'", "0.11.0", message)
                    # 若你想中断请解除注释上行
                    generator = None
            except Exception:
                generator = None

        # Set the sampling time (scheduler API assumed similar to HF diffusers)
        # Some schedulers expect set_timesteps() to be called with num_inference_steps
        try:
            self.scheduler.set_timesteps(num_inference_steps)
        except Exception as e:
            # 如果 scheduler 没有该方法，报错提示更明确
            raise RuntimeError(f"Scheduler does not support set_timesteps(num_inference_steps): {e}")

        # Generate random noise
        noise = torch.randn(image.shape, device=device, dtype=dtype)

        model_output = None
        middle_feat = None

        key = self.sample_key

        # DDIM-style (we treat any 'ddim' in key as DDIM)
        if 'ddim' in key:
            # 随机 timestep（每样本不同），注意 scheduler.add_noise 的签名应匹配你实现
            timesteps = torch.randint(0, getattr(self.scheduler, "num_train_timesteps", 1000), (batch_size,)).long().to(
                device)
            # add_noise 可能接受不同签名，这里按你原实现调用
            image = self.scheduler.add_noise(image, noise, timesteps).to(device)
            noise_image = image
            # iterate through scheduler.timesteps (assumed tensor/iterable)
            for t in self.scheduler.timesteps:
                model_output, middle_feat = self.model(image, t, device)
                # step 返回 dict，取 prev_sample
                step_out = self.scheduler.step(model_output, t, image, eta=eta, use_clipped_model_output=True,
                                               generator=generator)
                image = step_out['prev_sample']

        # dpmsolver / dpmsolver++ / deis / unipc / pndm （多步 solver）
        elif 'dpmsolver' in key or 'ddp-solver' in key or 'deis' in key or 'unipc' in key or 'pndm' in key:
            # 有些 scheduler 有 order 属性
            timesteps_full = getattr(self.scheduler, "timesteps", None)
            order = getattr(self.scheduler, "order", 0)
            if timesteps_full is None:
                raise RuntimeError("Scheduler.timesteps is not available for this scheduler.")
            timesteps = timesteps_full[order:].to(device)
            image = self.scheduler.add_noise(image, noise, timesteps[:1]).to(device)
            noise_image = image

            for t in timesteps_full:
                model_output, middle_feat = self.model(image, t, device)
                step_out = self.scheduler.step(model_output, t, image)
                image = step_out['prev_sample']

        # Heun
        elif 'heun' in key:
            timesteps_full = getattr(self.scheduler, "timesteps", None)
            order = getattr(self.scheduler, "order", 0)
            if timesteps_full is None:
                raise RuntimeError("Scheduler.timesteps is not available for this scheduler.")
            timesteps = timesteps_full[order:].to(device)
            image = self.scheduler.add_noise(image, noise, timesteps[:1]).to(device)
            noise_image = image
            for t in timesteps_full:
                image = self.scheduler.scale_model_input(image, t)
                model_output, middle_feat = self.model(image, t, device)
                image = self.scheduler.step(model_output, t, image)['prev_sample']

        # LMS
        elif 'lms' in key:
            timesteps_full = getattr(self.scheduler, "timesteps", None)
            order = getattr(self.scheduler, "order", 0)
            if timesteps_full is None:
                raise RuntimeError("Scheduler.timesteps is not available for this scheduler.")
            timesteps = timesteps_full[order:].to(device)
            image = self.scheduler.add_noise(image, noise, timesteps[:1]).to(device)
            noise_image = image
            for t in timesteps_full:
                image = self.scheduler.scale_model_input(image, t)
                model_output, middle_feat = self.model(image, t, device)
                image = self.scheduler.step(model_output, t, image)['prev_sample']

        # Euler
        elif 'euler' in key:
            timesteps_full = getattr(self.scheduler, "timesteps", None)
            order = getattr(self.scheduler, "order", 0)
            if timesteps_full is None:
                raise RuntimeError("Scheduler.timesteps is not available for this scheduler.")
            image = self.scheduler.add_noise(image, noise, timesteps_full[:1]).to(device)
            noise_image = image
            # 固定随机种子示例（可删）
            generator = torch.manual_seed(0)
            for t in timesteps_full:
                image = self.scheduler.scale_model_input(image, t)
                model_output, middle_feat = self.model(image, t, device)
                image = self.scheduler.step(model_output, t, image, generator=generator)['prev_sample']

        else:
            # 未识别的采样器，尝试用默认策略（类似 ddpm）
            timesteps_full = getattr(self.scheduler, "timesteps", None)
            if timesteps_full is None:
                raise RuntimeError("Unknown sampling strategy and scheduler.timesteps not found.")
            image = self.scheduler.add_noise(image, noise, timesteps_full[:1]).to(device)
            noise_image = image
            for t in timesteps_full:
                model_output, middle_feat = self.model(image, t, device)
                image = self.scheduler.step(model_output, t, image)['prev_sample']

        # 返回值：保持与上层调用一致
        if not return_dict:
            # (image, noise, model_output, middle_feat)
            return (image, noise, model_output, middle_feat)

        # 字典返回时包含更多信息，避免上层缺失
        return {'images': image, 'noise': noise, 'model_output': model_output, 'middle_feat': middle_feat}