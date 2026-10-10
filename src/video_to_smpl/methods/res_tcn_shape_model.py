from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.5):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve temporal length")
        self.block = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class AttentionPool1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=-1)
        return torch.sum(x * weights, dim=-1)


class ResTCNShapeRegressor(nn.Module):
    def __init__(
        self,
        num_joints: int = 17,
        in_dim: int = 3,
        hidden_dim: int = 256,
        num_blocks: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.5,
        head_dropout: float | None = None,
        mlp_hidden_dim: int = 128,
        pooling: str = "avg",
        num_betas: int = 10,
    ):
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if pooling not in {"avg", "attention"}:
            raise ValueError("pooling must be 'avg' or 'attention'")
        if head_dropout is None:
            head_dropout = dropout

        input_dim = int(num_joints) * int(in_dim)
        self.input_conv = nn.Conv1d(input_dim, int(hidden_dim), kernel_size=kernel_size, padding=kernel_size // 2)
        self.res_blocks = nn.Sequential(
            *[
                ResidualTemporalBlock(
                    channels=int(hidden_dim),
                    kernel_size=int(kernel_size),
                    dropout=float(dropout),
                )
                for _ in range(int(num_blocks))
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1) if pooling == "avg" else AttentionPool1d(int(hidden_dim))
        self.head = nn.Sequential(
            nn.Linear(int(hidden_dim), int(mlp_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(head_dropout)),
            nn.Linear(int(mlp_hidden_dim), int(num_betas)),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [B, T, J, 3]
        batch_size, time_steps, joints, channels = coords.shape
        x = coords.reshape(batch_size, time_steps, joints * channels).transpose(1, 2)
        x = self.input_conv(x)
        x = self.res_blocks(x)
        if isinstance(self.pool, nn.AdaptiveAvgPool1d):
            x = self.pool(x).squeeze(-1)
        else:
            x = self.pool(x)
        return self.head(x)


class TransitionResidualTemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve temporal length")
        self.block = nn.Sequential(
            nn.BatchNorm1d(int(in_channels)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Conv1d(
                int(in_channels),
                int(out_channels),
                kernel_size=int(kernel_size),
                padding=int(kernel_size) // 2,
            ),
        )
        self.skip = (
            nn.Identity()
            if int(in_channels) == int(out_channels)
            else nn.Conv1d(int(in_channels), int(out_channels), kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.block(x)


class StagedResTCNShapeRegressor(nn.Module):
    def __init__(
        self,
        num_joints: int = 17,
        in_dim: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.5,
        head_dropout: float | None = None,
        mlp_hidden_dim: int = 128,
        pooling: str = "avg",
        num_betas: int = 10,
    ):
        super().__init__()
        if pooling not in {"avg", "attention"}:
            raise ValueError("pooling must be 'avg' or 'attention'")
        if head_dropout is None:
            head_dropout = dropout

        input_dim = int(num_joints) * int(in_dim)
        self.input_conv = nn.Conv1d(input_dim, 128, kernel_size=kernel_size, padding=kernel_size // 2)
        self.stage1 = nn.Sequential(
            ResidualTemporalBlock(128, kernel_size=kernel_size, dropout=dropout),
            ResidualTemporalBlock(128, kernel_size=kernel_size, dropout=dropout),
        )
        self.transition1 = TransitionResidualTemporalBlock(128, 192, kernel_size=kernel_size, dropout=dropout)
        self.stage2 = nn.Sequential(
            ResidualTemporalBlock(192, kernel_size=kernel_size, dropout=dropout),
            ResidualTemporalBlock(192, kernel_size=kernel_size, dropout=dropout),
        )
        self.transition2 = TransitionResidualTemporalBlock(192, 256, kernel_size=kernel_size, dropout=dropout)
        self.stage3 = nn.Sequential(
            ResidualTemporalBlock(256, kernel_size=kernel_size, dropout=dropout),
            ResidualTemporalBlock(256, kernel_size=kernel_size, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1) if pooling == "avg" else AttentionPool1d(256)
        self.head = nn.Sequential(
            nn.Linear(256, int(mlp_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(head_dropout)),
            nn.Linear(int(mlp_hidden_dim), int(num_betas)),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, joints, channels = coords.shape
        x = coords.reshape(batch_size, time_steps, joints * channels).transpose(1, 2)
        x = self.input_conv(x)
        x = self.stage1(x)
        x = self.transition1(x)
        x = self.stage2(x)
        x = self.transition2(x)
        x = self.stage3(x)
        if isinstance(self.pool, nn.AdaptiveAvgPool1d):
            x = self.pool(x).squeeze(-1)
        else:
            x = self.pool(x)
        return self.head(x)


class PaperStyleResidualTemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 8,
        stride: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(int(in_channels)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Conv1d(
                int(in_channels),
                int(out_channels),
                kernel_size=int(kernel_size),
                stride=int(stride),
                padding=int(kernel_size) // 2,
            ),
        )
        self.skip = (
            nn.Identity()
            if int(in_channels) == int(out_channels) and int(stride) == 1
            else nn.Conv1d(int(in_channels), int(out_channels), kernel_size=1, stride=int(stride))
        )

    @staticmethod
    def _match_temporal_length(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == ref.shape[-1]:
            return x
        if x.shape[-1] > ref.shape[-1]:
            return x[..., : ref.shape[-1]]
        pad = ref.shape[-1] - x.shape[-1]
        return nn.functional.pad(x, (0, pad))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.block(x)
        residual = self._match_temporal_length(residual, out)
        return residual + out


class PaperStyleResTCNShapeRegressor(nn.Module):
    def __init__(
        self,
        num_joints: int = 17,
        in_dim: int = 3,
        kernel_size: int = 8,
        dropout: float = 0.5,
        head_dropout: float | None = None,
        mlp_hidden_dim: int = 128,
        pooling: str = "avg",
        num_betas: int = 10,
    ):
        super().__init__()
        if pooling not in {"avg", "attention"}:
            raise ValueError("pooling must be 'avg' or 'attention'")
        if head_dropout is None:
            head_dropout = dropout

        input_dim = int(num_joints) * int(in_dim)
        self.input_conv = nn.Conv1d(
            input_dim,
            64,
            kernel_size=int(kernel_size),
            padding=int(kernel_size) // 2,
        )
        self.block_a = nn.Sequential(
            PaperStyleResidualTemporalBlock(64, 64, kernel_size=kernel_size, stride=1, dropout=dropout),
            PaperStyleResidualTemporalBlock(64, 64, kernel_size=kernel_size, stride=1, dropout=dropout),
            PaperStyleResidualTemporalBlock(64, 64, kernel_size=kernel_size, stride=1, dropout=dropout),
        )
        self.block_b = nn.Sequential(
            PaperStyleResidualTemporalBlock(64, 128, kernel_size=kernel_size, stride=2, dropout=dropout),
            PaperStyleResidualTemporalBlock(128, 128, kernel_size=kernel_size, stride=1, dropout=dropout),
            PaperStyleResidualTemporalBlock(128, 128, kernel_size=kernel_size, stride=1, dropout=dropout),
        )
        self.block_c = nn.Sequential(
            PaperStyleResidualTemporalBlock(128, 256, kernel_size=kernel_size, stride=2, dropout=dropout),
            PaperStyleResidualTemporalBlock(256, 256, kernel_size=kernel_size, stride=1, dropout=dropout),
            PaperStyleResidualTemporalBlock(256, 256, kernel_size=kernel_size, stride=1, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1) if pooling == "avg" else AttentionPool1d(256)
        self.head = nn.Sequential(
            nn.Linear(256, int(mlp_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(head_dropout)),
            nn.Linear(int(mlp_hidden_dim), int(num_betas)),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [B, T, J, 3]
        batch_size, time_steps, joints, channels = coords.shape
        x = coords.reshape(batch_size, time_steps, joints * channels).transpose(1, 2)
        x = self.input_conv(x)
        x = self.block_a(x)
        x = self.block_b(x)
        x = self.block_c(x)
        if isinstance(self.pool, nn.AdaptiveAvgPool1d):
            x = self.pool(x).squeeze(-1)
        else:
            x = self.pool(x)
        return self.head(x)


class DilatedResidualTemporalBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve temporal length")
        padding = (int(kernel_size) // 2) * int(dilation)
        self.block = nn.Sequential(
            nn.BatchNorm1d(int(channels)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Conv1d(
                int(channels),
                int(channels),
                kernel_size=int(kernel_size),
                padding=padding,
                dilation=int(dilation),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class MotionFeatureDilatedTCNShapeRegressor(nn.Module):
    def __init__(
        self,
        num_joints: int = 17,
        in_dim: int = 3,
        feature_dim: int = 128,
        kernel_size: int = 5,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.5,
        num_betas: int = 10,
    ):
        super().__init__()
        input_dim = int(num_joints) * int(in_dim) * 3
        self.encoder = nn.Linear(input_dim, int(feature_dim))
        self.blocks = nn.Sequential(
            *[
                DilatedResidualTemporalBlock(
                    channels=int(feature_dim),
                    kernel_size=int(kernel_size),
                    dilation=int(dilation),
                    dropout=float(dropout),
                )
                for dilation in dilations
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(int(feature_dim) * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(128, int(num_betas)),
        )

    @staticmethod
    def _motion_features(coords: torch.Tensor) -> torch.Tensor:
        velocity = coords.new_zeros(coords.shape)
        velocity[:, 1:] = coords[:, 1:] - coords[:, :-1]
        acceleration = coords.new_zeros(coords.shape)
        acceleration[:, 1:] = velocity[:, 1:] - velocity[:, :-1]
        features = torch.cat([coords, velocity, acceleration], dim=-1)
        batch_size, time_steps, joints, channels = features.shape
        return features.reshape(batch_size, time_steps, joints * channels)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [B, T, J, 3]
        x = self._motion_features(coords)
        x = self.encoder(x).transpose(1, 2)
        x = self.blocks(x)
        avg_pool = torch.mean(x, dim=-1)
        max_pool = torch.amax(x, dim=-1)
        return self.head(torch.cat([avg_pool, max_pool], dim=1))


class PaperStyleDilatedResidualTemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 8,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        padding = ((int(kernel_size) - 1) * int(dilation)) // 2
        self.block = nn.Sequential(
            nn.BatchNorm1d(int(in_channels)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Conv1d(
                int(in_channels),
                int(out_channels),
                kernel_size=int(kernel_size),
                stride=int(stride),
                dilation=int(dilation),
                padding=padding,
            ),
        )
        self.skip = (
            nn.Identity()
            if int(in_channels) == int(out_channels) and int(stride) == 1
            else nn.Conv1d(int(in_channels), int(out_channels), kernel_size=1, stride=int(stride))
        )

    @staticmethod
    def _match_temporal_length(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == ref.shape[-1]:
            return x
        if x.shape[-1] > ref.shape[-1]:
            return x[..., : ref.shape[-1]]
        pad = ref.shape[-1] - x.shape[-1]
        return nn.functional.pad(x, (0, pad))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.block(x)
        residual = self._match_temporal_length(residual, out)
        return residual + out


class PaperStyleDilatedResTCNShapeRegressor(nn.Module):
    def __init__(
        self,
        num_joints: int = 17,
        in_dim: int = 3,
        kernel_size: int = 8,
        dropout: float = 0.5,
        mlp_hidden_dim: int = 128,
        pooling: str = "avg",
        num_betas: int = 10,
    ):
        super().__init__()
        if pooling not in {"avg", "attention"}:
            raise ValueError("pooling must be 'avg' or 'attention'")

        input_dim = int(num_joints) * int(in_dim)
        self.input_conv = nn.Conv1d(
            input_dim,
            64,
            kernel_size=int(kernel_size),
            padding=int(kernel_size) // 2,
        )
        self.block_a = nn.Sequential(
            PaperStyleDilatedResidualTemporalBlock(64, 64, kernel_size=kernel_size, stride=1, dilation=1, dropout=dropout),
            PaperStyleDilatedResidualTemporalBlock(64, 64, kernel_size=kernel_size, stride=1, dilation=2, dropout=dropout),
            PaperStyleDilatedResidualTemporalBlock(64, 64, kernel_size=kernel_size, stride=1, dilation=4, dropout=dropout),
        )
        self.block_b = nn.Sequential(
            PaperStyleDilatedResidualTemporalBlock(64, 128, kernel_size=kernel_size, stride=2, dilation=1, dropout=dropout),
            PaperStyleDilatedResidualTemporalBlock(128, 128, kernel_size=kernel_size, stride=1, dilation=2, dropout=dropout),
            PaperStyleDilatedResidualTemporalBlock(128, 128, kernel_size=kernel_size, stride=1, dilation=4, dropout=dropout),
        )
        self.block_c = nn.Sequential(
            PaperStyleDilatedResidualTemporalBlock(128, 256, kernel_size=kernel_size, stride=2, dilation=1, dropout=dropout),
            PaperStyleDilatedResidualTemporalBlock(256, 256, kernel_size=kernel_size, stride=1, dilation=4, dropout=dropout),
            PaperStyleDilatedResidualTemporalBlock(256, 256, kernel_size=kernel_size, stride=1, dilation=8, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1) if pooling == "avg" else AttentionPool1d(256)
        self.head = nn.Sequential(
            nn.Linear(256, int(mlp_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(mlp_hidden_dim), int(num_betas)),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, joints, channels = coords.shape
        x = coords.reshape(batch_size, time_steps, joints * channels).transpose(1, 2)
        x = self.input_conv(x)
        x = self.block_a(x)
        x = self.block_b(x)
        x = self.block_c(x)
        if isinstance(self.pool, nn.AdaptiveAvgPool1d):
            x = self.pool(x).squeeze(-1)
        else:
            x = self.pool(x)
        return self.head(x)


def _make_h36m17_edges() -> list[tuple[int, int]]:
    return [
        (0, 1), (1, 2), (2, 3),
        (0, 4), (4, 5), (5, 6),
        (0, 7), (7, 8), (8, 9), (9, 10),
        (8, 11), (11, 12), (12, 13),
        (8, 14), (14, 15), (15, 16),
    ]


def _make_stgcn_adjacency(num_joints: int = 17, center: int = 8) -> torch.Tensor:
    edges = _make_h36m17_edges()
    adjacency = torch.zeros((num_joints, num_joints), dtype=torch.float32)
    for i, j in edges:
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0

    hop = torch.full((num_joints, num_joints), fill_value=1e9, dtype=torch.float32)
    for node in range(num_joints):
        hop[node, node] = 0.0
    for i, j in edges:
        hop[i, j] = 1.0
        hop[j, i] = 1.0
    for k in range(num_joints):
        hop = torch.minimum(hop, hop[:, k:k + 1] + hop[k:k + 1, :])

    center_dist = hop[:, center]
    subsets = torch.zeros((3, num_joints, num_joints), dtype=torch.float32)
    for src in range(num_joints):
        for dst in range(num_joints):
            if hop[src, dst] > 1.0:
                continue
            if src == dst:
                subsets[0, dst, src] = 1.0
            elif center_dist[src] > center_dist[dst]:
                subsets[1, dst, src] = 1.0
            else:
                subsets[2, dst, src] = 1.0

    for part in range(subsets.shape[0]):
        degree = subsets[part].sum(dim=1, keepdim=True)
        degree = torch.where(degree > 0, degree, torch.ones_like(degree))
        subsets[part] = subsets[part] / degree
    return subsets


class STGCNUnit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
        temporal_kernel_size: int = 9,
        stride: int = 1,
        dropout: float = 0.5,
        residual: bool = True,
    ):
        super().__init__()
        self.kernel_size = int(adjacency.shape[0])
        self.register_buffer("adjacency", adjacency.clone())
        pad_t = (int(temporal_kernel_size) - 1) // 2
        self.gcn = nn.Conv2d(
            int(in_channels),
            int(out_channels) * self.kernel_size,
            kernel_size=(1, 1),
        )
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(int(out_channels)),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                int(out_channels),
                int(out_channels),
                kernel_size=(int(temporal_kernel_size), 1),
                stride=(int(stride), 1),
                padding=(pad_t, 0),
            ),
            nn.BatchNorm2d(int(out_channels)),
            nn.Dropout(float(dropout), inplace=False),
        )
        if not residual:
            self.residual = lambda x: 0
        elif int(in_channels) == int(out_channels) and int(stride) == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, stride=(int(stride), 1)),
                nn.BatchNorm2d(int(out_channels)),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.gcn(x)
        n, kc, t, v = x.shape
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum("nkctv,kvw->nctw", x, self.adjacency)
        x = self.tcn(x) + res
        return self.relu(x)


class STGCNShapeRegressor(nn.Module):
    def __init__(
        self,
        num_joints: int = 17,
        in_dim: int = 3,
        temporal_kernel_size: int = 9,
        dropout: float = 0.5,
        mlp_hidden_dim: int = 128,
        num_betas: int = 10,
    ):
        super().__init__()
        if int(num_joints) != 17 or int(in_dim) != 3:
            raise ValueError("STGCNShapeRegressor currently expects 17 joints with xyz input.")
        adjacency = _make_stgcn_adjacency(num_joints=17, center=8)
        self.data_bn = nn.BatchNorm1d(int(in_dim) * int(num_joints))
        self.layers = nn.ModuleList([
            STGCNUnit(3, 64, adjacency, temporal_kernel_size=temporal_kernel_size, stride=1, dropout=dropout, residual=False),
            STGCNUnit(64, 64, adjacency, temporal_kernel_size=temporal_kernel_size, stride=1, dropout=dropout),
            STGCNUnit(64, 64, adjacency, temporal_kernel_size=temporal_kernel_size, stride=1, dropout=dropout),
            STGCNUnit(64, 128, adjacency, temporal_kernel_size=temporal_kernel_size, stride=2, dropout=dropout),
            STGCNUnit(128, 128, adjacency, temporal_kernel_size=temporal_kernel_size, stride=1, dropout=dropout),
            STGCNUnit(128, 128, adjacency, temporal_kernel_size=temporal_kernel_size, stride=1, dropout=dropout),
            STGCNUnit(128, 256, adjacency, temporal_kernel_size=temporal_kernel_size, stride=2, dropout=dropout),
            STGCNUnit(256, 256, adjacency, temporal_kernel_size=temporal_kernel_size, stride=1, dropout=dropout),
            STGCNUnit(256, 256, adjacency, temporal_kernel_size=temporal_kernel_size, stride=1, dropout=dropout),
        ])
        self.head = nn.Sequential(
            nn.Linear(256, int(mlp_hidden_dim)),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(int(mlp_hidden_dim), int(num_betas)),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [B, T, J, 3]
        batch_size, time_steps, joints, channels = coords.shape
        x = coords.permute(0, 3, 1, 2).contiguous()  # [B, C, T, V]
        x = x.view(batch_size, channels * joints, time_steps)
        x = self.data_bn(x)
        x = x.view(batch_size, channels, joints, time_steps).permute(0, 1, 3, 2).contiguous()
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=-1).mean(dim=-1)
        return self.head(x)
