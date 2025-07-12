#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
from pathlib import Path
from omegaconf import OmegaConf
import torch
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    ProjectConfiguration,
    set_seed,
)
from tqdm.auto import tqdm
from torchvision.utils import save_image
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import is_compiled_module
import torch.nn.functional as F
import warnings
from dataloaders.my_dataset import PairedDataset, degradation_proc

warnings.filterwarnings("ignore")

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default="cfg.yml",
        help="path to config",
    )
    args = parser.parse_args()

    return args.config


def main():
    args = OmegaConf.load(parse_args())
    config = OmegaConf.load(args.degra_cfg)
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            OmegaConf.save(args, os.path.join(args.output_dir, "cfg.yml"))

    scheduler = DDPMScheduler.from_pretrained(args.sd_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.sd_path, subfolder="vae")
    vae.requires_grad_(False)
    vae.to(accelerator.device)

    from resnet import resnet34

    timestep_adaptive = resnet34(
        timestep_list=list(range(995, 5 - 1, -50)),
    )

    logger.info(
        f"Total timestep_adaptive training parameters: {sum([p.numel() for p in timestep_adaptive.parameters() if p.requires_grad]) / 1000000} M"
    )
    timestep_adaptive_opt = list(
        filter(lambda p: p.requires_grad, timestep_adaptive.parameters())
    )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        timestep_adaptive_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = PairedDataset(config.train)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    (
        timestep_adaptive,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        timestep_adaptive,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def kl_divergence(target_mu, target_var, x_src_mu, x_src_var):
        """
        Compute KL divergence between two Gaussian distributions.

        Args:
            target_mu: Mean of target distribution.
            target_var: Variance of target distribution.
            x_src_mu: Mean of source distribution.
            x_src_var: Variance of source distribution.

        Returns:
            KL divergence between target and source distributions.
        """
        # Ensure variances are positive to avoid numerical issues
        target_var = torch.clamp(target_var, min=1e-8)
        x_src_var = torch.clamp(x_src_var, min=1e-8)

        # Compute KL divergence
        kl = 0.5 * (
            torch.log(target_var / x_src_var)
            - 1
            + (x_src_var + (x_src_mu - target_mu) ** 2) / target_var
        )

        return kl

    for epoch in range(first_epoch, args.num_train_epochs):
        timestep_adaptive.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(*[timestep_adaptive]):
                # Prepare data
                x_src, x_tgt = degradation_proc(config, batch, accelerator.device)

                x_src_dist = vae.encode(x_src).latent_dist
                x_src_mu, x_src_var = x_src_dist.mean, x_src_dist.var
                x_tgt_dist = vae.encode(x_tgt).latent_dist
                x_tgt_mu, x_tgt_var = x_tgt_dist.mean, x_tgt_dist.var

                sqrt_alpha_cumprod_t = math.sqrt(
                    scheduler.alphas_cumprod[args.mid_timestep]
                )
                sqrt_one_minus_alpha_cumprod_t = math.sqrt(
                    1 - scheduler.alphas_cumprod[args.mid_timestep]
                )

                target_mu = sqrt_alpha_cumprod_t * x_tgt_mu
                target_var = (
                    sqrt_alpha_cumprod_t**2
                ) * x_tgt_var + sqrt_one_minus_alpha_cumprod_t**2

                loss = kl_divergence(target_mu, target_var, x_src_mu, x_src_var)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        timestep_adaptive_opt, args.max_grad_norm
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if (
                    accelerator.is_main_process
                    or accelerator.distributed_type == DistributedType.DEEPSPEED
                ):
                    if global_step % args.checkpointing_steps == 0:
                        # save_path = os.path.join(
                        #     args.output_dir, f"checkpoint-{global_step}"
                        # )
                        weight_path = os.path.join(
                            args.output_dir, f"weight-{global_step}"
                        )
                        # os.makedirs(save_path, exist_ok=True)
                        os.makedirs(weight_path, exist_ok=True)
                        # accelerator.save_state(save_path)
                        # logger.info(f"Saved state to {save_path}")
                        torch.save(
                            timestep_adaptive_opt.state_dict(),
                            os.path.join(weight_path, "timestep_adaptive_weights.pth"),
                        )
                        logger.info(f"Saved weight to {weight_path}")

            logs = {
                "loss": loss.detach().item(),
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # save_path = os.path.join(args.output_dir, f"weight-{global_step}")
        weight_path = os.path.join(args.output_dir, f"weight-{global_step}")
        # os.makedirs(save_path, exist_ok=True)
        os.makedirs(weight_path, exist_ok=True)
        torch.save(
            timestep_adaptive_opt.state_dict(),
            os.path.join(weight_path, "timestep_adaptive_weights.pth"),
        )
        logger.info(f"Saved weight to {weight_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
