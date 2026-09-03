# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm
import einops

from models.utils.continual_model import ContinualModel
from utils.args import (ArgumentParser, add_experiment_args,
                        add_management_args, add_rehearsal_args)

from utils.buffer import Buffer
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import time


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    parser.add_argument('--alpha', type=float, required=True, help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True, help='Penalty weight.')
    parser.add_argument('--masking_ratio', type=float, required=True)
    parser.add_argument('--gamma', type=float, required=True, help='Reconstruction-based regularization weight.')
    parser.add_argument('--linear_probing_epochs', type=int, default=0,
                        help='Number of linear probing epochs to perform at the end of each task.')
    parser.add_argument('--sampling_interval', type=int, default=5, help='Interval between frames.')
    parser.add_argument('--beta_dkd', type=float, default=1.6, help='DKD loss weight for old tasks.')

    # FreRA参数
    parser.add_argument('--use_frera', type=int, default=1, choices=[0, 1],
                        help='Whether to use FreRA frequency refinement (0: disable, 1: enable)')
    parser.add_argument('--frera_prob', type=float, default=0.5,
                        help='Probability to apply FreRA augmentation')
    parser.add_argument('--frera_strength', type=float, default=0.1,
                        help='Strength of FreRA augmentation')
    parser.add_argument('--frera_lambda', type=float, default=0.05,
                        help='Regularization weight for FreRA')
    parser.add_argument('--frera_warmup_epochs', type=int, default=3,
                        help='Number of epochs to wait before applying FreRA augmentation')

    return parser

NTU60_BONES_V2 = [
    (0, 1), (1, 20), (20, 2), (20, 4), (20, 8),
    (2, 3), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
    (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19)
]
BONES = NTU60_BONES_V2


def visualize_skeleton_from_tensor(x_tensor, sample_idx=0, num_frames=15, output_dir='fre_aug_vis',
                                   plane='xy', dpi=150, prefix='augmented', suffix=None):

    os.makedirs(output_dir, exist_ok=True)

    data = x_tensor[sample_idx].detach().cpu().numpy()  # (C, T, V)
    C, T, V = data.shape
    if V != 25:
        print(f"[Vis] Warning: V={V}, expected 25. Will still try to plot first 25 nodes if V>25.")
    V_plot = min(V, 25)

    if C >= 2:
        x_coords = data[0, :, :V_plot]
        y_coords = data[1, :, :V_plot]
    else:
        raise ValueError("Expect at least 2 channels for (x, y) coordinates")

    frames_idx = np.linspace(0, T - 1, num_frames, dtype=int)

    if suffix is not None:
        base_filename = f"{prefix}_{suffix}"
    else:
        base_filename = prefix

    for idx in frames_idx:
        x_frame = x_coords[idx]
        y_frame = y_coords[idx]

        fig, ax = plt.subplots(figsize=(6, 6))
        for start, end in BONES:
            if start < V_plot and end < V_plot:
                ax.plot([x_frame[start], x_frame[end]], [y_frame[start], y_frame[end]],
                        'b-', linewidth=2)
        ax.scatter(x_frame, y_frame, c='r', s=30, marker='o')

        x_min, x_max = x_frame.min(), x_frame.max()
        y_min, y_max = y_frame.min(), y_frame.max()
        if x_max - x_min < 1e-6:
            x_min, x_max = x_min - 0.1, x_max + 0.1
        if y_max - y_min < 1e-6:
            y_min, y_max = y_min - 0.1, y_max + 0.1
        size = max(x_max - x_min, y_max - y_min)
        margin = 0.1 * size
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Sample {sample_idx} – {base_filename}_frame{idx:04d}')
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.5)

        save_path = os.path.join(output_dir, f'{base_filename}_frame_{idx:04d}.png')
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    print(f"[Vis] Saved {num_frames} skeleton frames to {output_dir}/ with base name {base_filename}")


class FreRA(nn.Module):

    def __init__(self, seq_len, num_joints, num_channels=3, temperature=0.2):
        super().__init__()

        self.seq_len = seq_len
        self.num_joints = num_joints
        self.num_channels = num_channels
        self.temperature = temperature

        F_dim = seq_len // 2 + 1

        self.s = nn.Parameter(torch.zeros(F_dim))
        nn.init.normal_(self.s, mean=1.0, std=1.0)

    def apply_frequency_refinement(self, x):

        B = x.shape[0]

        two_skel_mask = None
        if x.dim() == 5:
            two_skel_mask = x[..., 1].sum((1, 2, 3)) > 0
            if not two_skel_mask.any():
                x_in = x[..., 0]
            else:
                x_in = torch.cat([x[..., 0], x[two_skel_mask, ..., 1]], dim=0)
        else:
            x_in = x

        len_parts = getattr(self, 'len_parts', 1)
        if hasattr(self, 'len_parts'):
            x_in = einops.rearrange(x_in, 'n c (t p) v -> n c t (p v)', p=len_parts)

        x_freq = torch.fft.rfft(x_in, dim=2, norm='ortho')  # (B, C, F, V)

        s_mean = torch.mean(self.s)
        threshold = torch.min(torch.tensor(0.0, device=x.device), s_mean)

        D_mask = (self.s < threshold).float()

        if self.training:
            eps = torch.rand_like(self.s).clamp(min=1e-8, max=1.0 - 1e-8)
            gumbel_noise = torch.log(eps) - torch.log(1 - eps)
            w_crit = torch.sigmoid((gumbel_noise + self.s) / self.temperature)
        else:
            w_crit = (self.s > 0.0).float()

        with torch.no_grad():

            importance = torch.sigmoid(self.s / self.temperature)
            w_dist = (2.0 - importance) * D_mask
            w_dist = torch.clamp(w_dist,max=3.0)

        w_crit = w_crit.view(1, 1, -1, 1)
        w_dist = w_dist.view(1, 1, -1, 1)

        x_aug_freq = x_freq * (w_crit + w_dist)

        x_aug = torch.fft.irfft(x_aug_freq, n=x_in.shape[2], dim=2, norm='ortho')

        if hasattr(self, 'len_parts'):
            x_aug = einops.rearrange(x_aug, 'n c t (p v) -> n c (t p) v', p=len_parts)

        if x.dim() == 5:
            x_aug_full = torch.zeros_like(x)
            x_aug_full[..., 0] = x_aug[:B]
            if two_skel_mask is not None and two_skel_mask.any():
                x_aug_full[two_skel_mask, ..., 1] = x_aug[B:]
            x_aug = x_aug_full
        elif x.dim() == 4 and x_aug.shape[0] != B:
            x_aug = x_aug[:B]

        reg_loss = torch.mean(w_crit) if self.training else torch.tensor(0.0, device=x.device)

        if self.training and torch.rand(1).item() < 0.001:
            with torch.no_grad():
                fl = self.s
                print(f"[FreRA Debug] s: shape={fl.shape}, "
                      f"mean={fl.mean().item():.4f}, std={fl.std().item():.4f}, "
                      f"min={fl.min().item():.4f}, max={fl.max().item():.4f}")

                imp = torch.sigmoid(fl)
                print(f"[FreRA Debug] importance: mean={imp.mean().item():.4f}, std={imp.std().item():.4f}, "
                      f"min={imp.min().item():.4f}, max={imp.max().item():.4f}")

                wc_flat = w_crit.flatten()
                print(f"[FreRA Debug] w_crit (>0.5 ratio): {(wc_flat > 0.5).float().mean().item():.4f}, "
                      f"mean={wc_flat.mean().item():.4f}, std={wc_flat.std().item():.4f}")

                wd_flat = w_dist.flatten()
                print(f"[FreRA Debug] w_dist: mean={wd_flat.mean().item():.4f}, std={wd_flat.std().item():.4f}, "
                      f"min={wd_flat.min().item():.4f}, max={wd_flat.max().item():.4f}")


        x_final = x_aug
        return x_final, reg_loss

    def forward(self, x):
        return self.apply_frequency_refinement(x)


class SFERR(ContinualModel):
    NAME = 'sferr'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(SFERR, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device, args=self.args)

        self.gamma = self.args.gamma

        self.use_frera = self.args.use_frera
        self.frera_prob = self.args.frera_prob
        self.frera_strength = self.args.frera_strength
        self.frera_lambda = self.args.frera_lambda
        self.frera_warmup_epochs = self.args.frera_warmup_epochs

        if self.use_frera and hasattr(self.net, 'num_frames') and hasattr(self.net, 'num_joints'):
            num_frames = self.net.num_frames
            num_joints = self.net.num_joints
            len_parts = getattr(self.net, 'len_parts', 1)

            effective_joints = num_joints
            num_frames = num_frames * len_parts

            self.frera_module = FreRA(
                seq_len=num_frames,
                num_joints=effective_joints,
                num_channels=getattr(self.net, 'num_channels', 3),
                temperature=1.0
            ).to(self.device)

            print(f"[FreRA] Adding FreRA parameters to optimizer...")
            print(f"[FreRA] FreRA parameters: {sum(p.numel() for p in self.frera_module.parameters())}")

            frera_lr = 0.01

            self.frera_optimizer = torch.optim.Adam(
                self.frera_module.parameters(),
                lr=frera_lr,
                weight_decay=0.001,
                betas=(0.9,0.999)
            )

            print(f"[FreRA] Added FreRA parameters to existing optimizer")
            print(f"[FreRA] Now optimizer has {len(self.opt.param_groups)} parameter groups")
        else:
            self.frera_module = None

        #######################################################################
        self.task_classes = []
        self.current_task = 0
        self.class_prototypes = {}
        self.class_feature_sums = {}
        self.class_sample_counts = {}
        self.features = {}
        self.epoch = 0
        self.visualize_count = 0
        self.class_feature_accumulators = {}
        self.class_feature_variance = {}
        self.classes_per_task = 10
        self.num_total_classes = 60

    def _reinitialize_optimizer_with_frera(self):
        all_params = []

        for name, param in self.net.named_parameters():
            if param.requires_grad:
                all_params.append(param)

        for name, param in self.frera_module.named_parameters():
            if param.requires_grad:
                all_params.append(param)

        optimizer_type = self.args.optimizer if hasattr(self.args, 'optimizer') else 'sgd'
        lr = 0.1
        weight_decay = self.args.weight_decay if hasattr(self.args, 'weight_decay') else 0.0001

        if optimizer_type.lower() == 'adam':
            self.opt = torch.optim.Adam(all_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            self.opt = torch.optim.SGD(all_params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            self.opt = torch.optim.Adam(all_params, lr=lr, weight_decay=weight_decay)

        print(f"[FreRA] Reinitialized optimizer with {len(all_params)} parameters")

    def apply_frera_augmentation(self, inputs,labels=None,**kwargs):

        frera_loss = torch.tensor(0.0, device=inputs.device)
        augmented_inputs_out = inputs

        if self.use_frera and self.training and self.epoch >= self.frera_warmup_epochs:
            augmented_inputs, module_reg_loss = self.frera_module(inputs)


            augmented_inputs_out = augmented_inputs

            frera_loss = self.frera_lambda * module_reg_loss

        return augmented_inputs_out, frera_loss

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()
        if hasattr(self, 'frera_optimizer'):
            self.frera_optimizer.zero_grad()

        tot_loss = 0

        if self.use_frera:
            augmented_inputs, frera_loss = self.apply_frera_augmentation(inputs, labels)
        else:
            augmented_inputs = inputs
            frera_loss = torch.tensor(0.0, device=inputs.device)

        outputs, reconstructions = self.net(augmented_inputs)
        loss_stream = self.loss(outputs, labels)
        loss_mae_stream = self.gamma * F.mse_loss(reconstructions, augmented_inputs)
        # self.gamma = 0.5

        # 添加FreRA正则化损失
        loss = loss_stream + loss_mae_stream + frera_loss
        tot_loss += loss.item()
        loss.backward()

        if not self.buffer.is_empty():

            buf_inputs, _, buf_logits, _ ,_= self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            buf_outputs, buf_reconstructions = self.net(buf_inputs)
            if self.current_task > 0:
                num_learned_classes = (self.current_task) * self.classes_per_task
                if num_learned_classes > 0:
                    mask = torch.zeros_like(buf_outputs)
                    mask[:, :num_learned_classes] = 1.0

                    masked_buf_outputs = buf_outputs * mask
                    masked_buf_logits = buf_logits * mask

                    loss_logits = F.mse_loss(masked_buf_outputs, masked_buf_logits)
                else:
                    loss_logits = torch.tensor(0.0, device=buf_outputs.device)
            else:
                loss_logits = torch.tensor(0.0, device=buf_outputs.device)
            loss_mae_logits = self.gamma * F.mse_loss(buf_reconstructions, buf_inputs)
            loss = self.args.alpha * (loss_logits + loss_mae_logits)
            # alpha = 0.3
            tot_loss += loss.item()
            if loss_logits != 0:
                loss.backward()

            buf_inputs, buf_labels, buf_logits, _ ,_= self.buffer.get_data(self.args.minibatch_size,
                                                                            transform=self.transform)
            buf_outputs, buf_reconstructions = self.net(buf_inputs)

            if self.current_task > 0:
                num_learned_classes = (self.current_task) * self.classes_per_task

                if num_learned_classes > 0:
                    mask = torch.zeros_like(buf_outputs)
                    mask[:, :num_learned_classes] = 1.0

                    temp_outputs = buf_outputs.clone()
                    temp_outputs[:, num_learned_classes:] = -1e10
                    loss_labels = self.loss(temp_outputs, buf_labels)
                else:
                    loss_labels = torch.tensor(0.0, device=buf_outputs.device)
            else:
                loss_labels = self.loss(buf_outputs, buf_labels)

            loss_mae_labels = self.gamma * F.mse_loss(buf_reconstructions, buf_inputs)
            loss = self.args.beta * (loss_labels + loss_mae_labels)
            # beta = 0.8
            tot_loss += loss.item()
            if loss_labels != 0:
                loss.backward()

        if self.args.grad_clip:
            torch.nn.utils.clip_grad_value_(self.net.parameters(), self.args.grad_clip)

        if self.use_frera and self.frera_module is not None:
            total_grad_norm = 0.0
            max_grad = 0.0
            for name, param in self.frera_module.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm
                    max_grad = max(max_grad, torch.abs(param.grad).max().item())

            if self.epoch > 3 and self.epoch % 5 == 0 and torch.rand(1).item() < 0.0001:
                print(f"[FreRA Grad] Total norm: {total_grad_norm:.6f}, Max grad: {max_grad:.6f}")
                if total_grad_norm < 1e-8:
                    print(f"[FreRA Warning] No gradients detected!")

            if hasattr(self, 'frera_optimizer'):
                if self.args.grad_clip:
                    torch.nn.utils.clip_grad_value_(self.frera_module.parameters(), self.args.grad_clip)
                if self.use_frera and self.frera_module is not None:
                    if hasattr(self, 'frera_optimizer'):
                        self.frera_optimizer.step()
        self.opt.step()

        # 重建损失
        if self.epoch == self.args.n_epochs - 1:
            with torch.no_grad():
                self.features = self.net.get_features(inputs) if hasattr(self.net, 'get_features') else None
                if hasattr(self.net, 'get_reconstruction'):
                    reconstructions = self.net.get_reconstruction(inputs)
                    recon_losses = F.mse_loss(reconstructions, inputs, reduction='none')
                    recon_losses = recon_losses.mean(dim=tuple(range(1, recon_losses.dim())))
                    # print(recon_losses)
                else:
                    recon_losses = None
                # print(recon_losses)
                self.buffer.add_data(
                    examples=not_aug_inputs,
                    labels=labels,
                    logits=outputs.data,
                    features=self.features.detach(),
                    recon_losses=recon_losses.detach()
                )

        return tot_loss

    def observe_lp(self, inputs, labels, not_aug_inputs, epoch):

        self.opt.zero_grad()

        tot_loss = 0

        outputs = self.net(inputs)
        loss_stream_lp = self.loss(outputs, labels)
        tot_loss += loss_stream_lp.item()
        loss_stream_lp.backward()

        if not self.buffer.is_empty():

            buf_inputs, _, buf_logits, _ ,_= self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss_logits_lp = F.mse_loss(buf_outputs, buf_logits)
            loss = self.args.alpha * loss_logits_lp
            tot_loss += loss.item()
            loss.backward()

            buf_inputs, buf_labels, _, _ ,_ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss_labels_lp = self.loss(buf_outputs, buf_labels)
            loss = self.args.beta * loss_labels_lp
            tot_loss += loss.item()
            loss.backward()

        if self.args.grad_clip:
            torch.nn.utils.clip_grad_value_(self.net.parameters(), self.args.grad_clip)

        self.opt.step()

        return tot_loss

    def begin_task(self, dataset):

        self.buffer.set_current_task(self.current_task)
        self.buffer.current_task_cache = {}

        self.visualize_count = 0

    def end_task(self, train_loader):

        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False

        if self.frera_module is not None:
            for param in self.frera_module.parameters():
                param.requires_grad = False

        for param in self.net.classifier.parameters():
            param.requires_grad = True

        for epoch in range(self.args.linear_probing_epochs):
            with tqdm(total=len(train_loader), desc=f"Linear probing on the full poses [Epoch: {epoch}]") as pbar:
                for i, data in enumerate(train_loader):
                    if self.args.debug_mode and i > 3:
                        break

                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.float().to(self.device), labels.to(self.device)
                    not_aug_inputs = not_aug_inputs.float().to(self.device)
                    loss = self.observe_lp(inputs, labels, not_aug_inputs, epoch)

                    assert not math.isnan(loss)

                    pbar.update()

        if self.buffer.mode == 'herding':
            self.buffer.select_prototypes(self.current_task)

        self.update_buffer_logits()
        for param in self.net.parameters():
            param.requires_grad = True

        if self.frera_module is not None:
            for param in self.frera_module.parameters():
                param.requires_grad = True

        self.net.train()
        self.current_task += 1

    @torch.no_grad()
    def update_buffer_logits(self):
        B = self.args.batch_size

        for i in range(0, len(self.buffer), B):
            idxs = torch.arange(i, min(i + B, len(self.buffer)))
            buf_inputs, *_ = self.buffer.get_data_by_index(idxs, transform=self.transform)
            buf_logits = self.net(buf_inputs)
            self.buffer.logits[idxs] = buf_logits.data

    def begin_epoch(self, epoch):
        self.epoch = epoch

        if self.use_frera and self.frera_module is not None and epoch % 10 == 0:
            if hasattr(self.frera_module, 'freq_logits'):
                importance = torch.sigmoid(self.frera_module.freq_logits)