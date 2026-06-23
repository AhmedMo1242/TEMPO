import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models

from .stgcn_layers import Graph, STGCN_block


def generate_mask(shape, part_num, clip_length, ratio, dim):
    """
    Optimized Complementary Masks Implementation using PyTorch
    """
    B, T, C = shape
    clips = T // clip_length

    # Move to device early if possible, or stay on CPU but use vectorized ops
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Vectorized random mask generation
    random_mask = torch.rand((B, clips, part_num), device=device) > (1 - 2 * ratio)

    mask_q = torch.zeros_like(random_mask)
    mask_k = torch.zeros_like(random_mask)

    # Get indices of active masks
    indices = torch.where(random_mask)
    num_active = indices[0].size(0)

    if num_active > 0:
        # Randomly split active indices into two halves
        perm = torch.randperm(num_active, device=device)
        half_num = num_active // 2

        q_indices = perm[:half_num]
        k_indices = perm[half_num:]

        mask_q[indices[0][q_indices], indices[1][q_indices], indices[2][q_indices]] = 1
        mask_k[indices[0][k_indices], indices[1][k_indices], indices[2][k_indices]] = 1

    # Efficiently expand masks to full (B, T, C) shape
    # Repeat along temporal dimension
    mask_q_full = mask_q.repeat_interleave(clip_length, dim=1)
    mask_k_full = mask_k.repeat_interleave(clip_length, dim=1)

    # Handle remainder frames if T is not divisible by clip_length
    if T % clip_length != 0:
        last_clip_q = mask_q[:, -1:, :].repeat_interleave(T % clip_length, dim=1)
        last_clip_k = mask_k[:, -1:, :].repeat_interleave(T % clip_length, dim=1)
        mask_q_full = torch.cat([mask_q_full, last_clip_q], dim=1)
        mask_k_full = torch.cat([mask_k_full, last_clip_k], dim=1)

    # Repeat along channel dimension (group-wise)
    mask_q_full = mask_q_full.repeat_interleave(dim, dim=2)
    mask_k_full = mask_k_full.repeat_interleave(dim, dim=2)

    # If original T was slightly different due to padding in repeat_interleave
    mask_q_full = mask_q_full[:, :T, :C]
    mask_k_full = mask_k_full[:, :T, :C]

    return 1.0 - mask_q_full.float(), 1.0 - mask_k_full.float()


class CoSign1s_block(nn.Module):
    def __init__(self, modes, indims, outdims, A, split, temporal_kernel, adaptive):
        super(CoSign1s_block, self).__init__()
        self.modes = modes
        self.indims = indims
        self.outdims = outdims
        self.A = A
        self.split = split
        self.temporal_kernel = temporal_kernel
        self.gcn_modules = {}
        self.spatial_kernel_size = A[0].size(0)
        self.adaptive = adaptive
        for index, mode in enumerate(self.modes):
            # Group-specific GCN
            self.gcn_modules[mode] = STGCN_block(
                indims,
                outdims,
                (self.temporal_kernel, self.spatial_kernel_size),
                A[index].clone(),
                self.adaptive,
            )
        self.gcn_modules = nn.ModuleDict(self.gcn_modules)

    def forward(self, feature):
        index = 0
        feat_list = []
        for mode in self.modes:
            if index == 0:
                start, end = 0, self.split[0]
            else:
                start, end = self.split[index - 1], self.split[index]
            if mode == "hand21":
                hand = self.gcn_modules[mode](
                    torch.cat(
                        [
                            feature[:, :, :, start:end],
                            feature[:, :, :, end : self.split[index + 1]],
                        ]
                    )
                )
                left, right = torch.chunk(hand, 2, dim=0)
                feat_list.append(left)
                feat_list.append(right)
                index += 2
            else:
                feat_list.append(self.gcn_modules[mode](feature[:, :, :, start:end]))
                index += 1
        return torch.cat(feat_list, dim=-1)


class CoSign2s(nn.Module):
    def __init__(
        self,
        in_channels,
        split,
        temporal_kernel,
        hidden_size,
        modes,
        level,
        adaptive=True,
        CR_args=None,
    ) -> None:
        super().__init__()
        self.split = split
        self.graph, A = {}, []
        self.part_num = len(self.split)
        self.in_channels = in_channels
        self.modes = modes
        self.CR_args = CR_args
        self.level = level
        for mode in self.modes:
            self.graph[mode] = Graph(
                layout=f"custom_{mode}", strategy="distance", max_hop=1
            )
            A.append(
                torch.tensor(
                    self.graph[mode].A, dtype=torch.float32, requires_grad=False
                )
            )

        # Stability: Input normalization to handle unnormalized skeleton data
        # Increased eps to 1e-3 to handle zero-variance frames (signer not moving)
        self.static_input_norm = nn.LayerNorm(in_channels, eps=1e-3)
        self.motion_input_norm = nn.LayerNorm(4, eps=1e-3)

        self.static_linear = nn.Sequential(
            nn.Linear(in_channels, 64), nn.LayerNorm(64), nn.ReLU(inplace=True)
        )
        self.motion_linear = nn.Sequential(
            nn.Linear(4, 64), nn.LayerNorm(64), nn.ReLU(inplace=True)
        )

        # Override motion linear if configured differently?
        # Actually visual_extractor.py uses in_channels*2 in original init,
        # but feeder provides 4 channels. Let's make it robust.

        self.layer_configs = {
            "0": {
                "static": [(64, 64), (64, 128), (128, 256)],
                "motion": [(64, 64), (64, 128), (128, 256)],
                "fusion": [(128, 128), (256, 256), (512, 512)],
            },
            "1": {
                "static": [(64, 64), (64, 64), (64, 128), (128, 128), (128, 256)],
                "motion": [(64, 64), (64, 64), (64, 128), (128, 128), (128, 256)],
                "fusion": [(128, 128), (128, 128), (256, 256), (256, 256), (512, 512)],
            },
        }

        self.create_layers(A, temporal_kernel, adaptive)

        self.fusion_fusion = nn.Sequential(
            nn.Linear(512 * self.part_num, hidden_size), nn.ReLU(inplace=True)
        )

        self.pool_func = F.avg_pool2d
        self.out_size = hidden_size
        self.final_dim_static = 256
        self.final_dim_motion = 256
        self.final_dim_fusion = 512

    def create_layers(self, A, temporal_kernel, adaptive):
        config = self.layer_configs[self.level]

        for layer_type, layer_dims in config.items():
            layers = nn.ModuleList()

            for i, (in_dim, out_dim) in enumerate(layer_dims):
                layer_name = self.get_layer_name(layer_type, i)

                layer = CoSign1s_block(
                    self.modes,
                    in_dim,
                    out_dim,
                    A,
                    self.split,
                    temporal_kernel,
                    adaptive,
                )
                layers.append(layer)
                setattr(self, layer_name, layer)

            setattr(self, f"{layer_type}_layers", layers)

    def get_layer_name(self, layer_type, index):
        if self.level == "0":
            return f"{layer_type}_layer{index + 1}"
        else:
            if index < 4:
                return f"{layer_type}_layer{index // 2 + 1}_{index % 2 + 1}"
            else:
                return f"{layer_type}_layer3"

    def pooling_stage(self, feature):
        feature_list = []
        for i in range(len(self.split)):
            if i == 0:
                start, end = 0, self.split[0]
            else:
                start, end = self.split[i - 1], self.split[i]
            feature_list.append(
                self.pool_func(feature[:, :, :, start:end], (1, end - start)).squeeze(
                    -1
                )
            )
        return torch.cat(feature_list, dim=1)

    def process_static_motion(self, static, motion):
        if self.level == "0":
            processing_steps = [
                {
                    "static_steps": [1],
                    "motion_steps": [1],
                    "fusion_steps": [1],
                    "fusion_input": "concat",
                },
                {
                    "static_steps": [1],
                    "motion_steps": [1],
                    "fusion_steps": [1],
                    "fusion_input": "concat_sum",
                },
                {
                    "static_steps": [1],
                    "motion_steps": [1],
                    "fusion_steps": [1],
                    "fusion_input": "concat_sum",
                },
            ]
        else:
            processing_steps = [
                {
                    "static_steps": [1, 1],
                    "motion_steps": [1, 1],
                    "fusion_steps": [1, 1],
                    "fusion_input": "concat",
                },
                {
                    "static_steps": [1, 1],
                    "motion_steps": [1, 1],
                    "fusion_steps": [1, 1],
                    "fusion_input": "concat_sum",
                },
                {
                    "static_steps": [1],
                    "motion_steps": [1],
                    "fusion_steps": [1],
                    "fusion_input": "concat_sum",
                },
            ]

        static_idx = 0
        motion_idx = 0
        fusion_idx = 0
        fusion = None

        for step in processing_steps:
            for _ in step["static_steps"]:
                static = self.static_layers[static_idx](static)
                static_idx += 1

            for _ in step["motion_steps"]:
                motion = self.motion_layers[motion_idx](motion)
                motion_idx += 1

            if step["fusion_input"] == "concat":
                fusion_input = torch.cat([static, motion], dim=1)
            else:
                # Balanced fusion with scale factor
                fusion_input = torch.cat([fusion, (static + motion) * 0.5], dim=1)

            # Grad guard
            fusion_input = torch.nan_to_num(fusion_input, 0.0)

            for _ in step["fusion_steps"]:
                fusion = self.fusion_layers[fusion_idx](fusion_input)
                fusion_input = fusion
                fusion_idx += 1

        return static, motion, fusion

    def apply_masks(self, cat_feat_static, cat_feat_motion, cat_feat_fusion):
        stream_configs = [
            ("static", cat_feat_static, self.final_dim_static),
            ("motion", cat_feat_motion, self.final_dim_motion),
            ("fusion", cat_feat_fusion, self.final_dim_fusion),
        ]

        results = {}
        for stream_type, cat_feat, final_dim in stream_configs:
            mask_view1, mask_view2 = generate_mask(
                cat_feat.shape,
                self.part_num,
                self.CR_args["clip_length"],
                self.CR_args["ratio"],
                final_dim,
            )
            view1 = mask_view1.to(cat_feat.device) * cat_feat
            view2 = mask_view2.to(cat_feat.device) * cat_feat

            if stream_type == "fusion":
                view1 = self.fusion_fusion(view1)
                view2 = self.fusion_fusion(view2)

            results[f"view1_{stream_type}"] = view1
            results[f"view2_{stream_type}"] = view2

        return results

    def forward(self, x, len_x):
        if x.shape[3] == 7:
            static = torch.cat([x[:, :, :, 0:2], x[:, :, :, 6].unsqueeze(-1)], dim=-1)
        else:
            static = x
        static = static[:, :, :, : self.in_channels]
        motion = x[:, :, :, 2:6]

        # Last resort safety: mask out any remaining NaNs
        # (can happen if input is entirely NaN in the source data)
        static = torch.nan_to_num(static, 0.0)
        motion = torch.nan_to_num(motion, 0.0)

        # Apply input normalization (per joint, per frame)
        static = self.static_input_norm(static)
        motion = self.motion_input_norm(motion)

        static = self.static_linear(static).permute(0, 3, 1, 2)  # N,C,T,V
        motion = self.motion_linear(motion).permute(0, 3, 1, 2)  # N,C,T,V

        static, motion, fusion = self.process_static_motion(static, motion)

        cat_feat_static = self.pooling_stage(static).transpose(1, 2)  # B,T,C
        cat_feat_motion = self.pooling_stage(motion).transpose(1, 2)
        cat_feat_fusion = self.pooling_stage(fusion).transpose(1, 2)

        if self.CR_args is not None and self.training:
            return self.apply_masks(cat_feat_static, cat_feat_motion, cat_feat_fusion)
        else:
            fusion_feat_fusion = self.fusion_fusion(cat_feat_fusion)
            return {
                "fusion": fusion_feat_fusion,
            }
