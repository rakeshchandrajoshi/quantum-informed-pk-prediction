"""Mechanism-factorized and edge-gated graph prediction architectures.

Dimensions and parameter counts follow the manuscript's specified architecture.
Chemical feature creation is intentionally separate: models consume prepared
molecular descriptors and atomic/bond feature tensors.
"""
from __future__ import annotations

from typing import Mapping, Sequence
import torch
from torch import Tensor, nn
from torch.nn import functional as F

FACTORS = ("size", "lipophilicity", "ionization", "polarity", "binding", "electronic", "structural")


class FactorProjector(nn.Module):
    def __init__(self, input_dim: int, *, gated: bool, dropout: float):
        super().__init__()
        self.value = nn.Linear(input_dim, 8)
        self.gate = nn.Linear(input_dim, 8) if gated else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        value = self.dropout(F.silu(self.value(x)))
        if self.gate is not None:
            value = value * torch.sigmoid(self.gate(x))
        return value


class MFMN(nn.Module):
    """Seven descriptor factors + rank-eight FM interaction + per-endpoint head.

    Seven auxiliary heads are registered for *all* endpoints, matching the paper's
    reported parameter counts. For Fu these heads are never used in forward/loss.
    In CL/VDss they are trained only when auxiliary labels are supplied.
    """
    def __init__(self, context_dim: int, factor_dims: Mapping[str, int], *,
                 endpoint: str, dropout: float | None = None):
        super().__init__()
        if endpoint not in {"CL", "VDss", "Fu"}:
            raise ValueError("MFMN supports CL, VDss, and Fu")
        if set(factor_dims) != set(FACTORS) or any(factor_dims[k] <= 0 for k in FACTORS):
            raise ValueError("factor_dims must contain exactly seven nonempty factors")
        self.endpoint = endpoint
        self.gated = endpoint in {"CL", "Fu"}
        drop = (0.35 if endpoint == "Fu" else 0.2) if dropout is None else dropout
        self.projectors = nn.ModuleDict({
            name: FactorProjector(context_dim + factor_dims[name], gated=self.gated, dropout=drop)
            for name in FACTORS
        })
        self.fm_projectors = nn.ModuleDict({name: nn.Linear(8, 8) for name in FACTORS})
        self.aux_heads = nn.ModuleDict({name: nn.Linear(8, 1) for name in FACTORS})
        self.head = nn.Sequential(nn.Linear(64, 16), nn.SiLU(), nn.Dropout(drop), nn.Linear(16, 1))

    def forward(self, context: Tensor, factors: Mapping[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        embeddings = {}
        for name in FACTORS:
            embeddings[name] = self.projectors[name](torch.cat([context, factors[name]], dim=-1))
        latent = torch.cat([embeddings[name] for name in FACTORS], dim=-1)  # [B,56]
        u = torch.stack([self.fm_projectors[name](embeddings[name]) for name in FACTORS], dim=1)
        # 0.5 * [(sum u)^2 - sum(u^2)] calculates all 21 factor pairs.
        interaction = 0.5 * (u.sum(dim=1).square() - u.square().sum(dim=1))
        prediction = self.head(torch.cat([latent, interaction], dim=-1)).squeeze(-1)
        aux = ({name: self.aux_heads[name](embeddings[name]).squeeze(-1) for name in FACTORS}
               if self.endpoint != "Fu" else {})
        return prediction, aux


class ConventionalMLP(nn.Module):
    """Parameter-matched, concatenated-descriptor comparison model.

    Recommended (not historically verified) widths: CL [80,64,32], VDss [48,32,16].
    No mechanism gates, interaction layer, or auxiliary supervision.
    """
    def __init__(self, input_dim: int, *, endpoint: str,
                 widths: Sequence[int] | None = None, dropout: float = 0.2):
        super().__init__()
        if endpoint not in {"CL", "VDss", "Fu"}:
            raise ValueError("MLP supports CL, VDss, and Fu")
        hidden = list(widths or ({"CL": (80, 64, 32), "VDss": (48, 32, 16),
                                  "Fu": (100, 64, 32)}[endpoint]))
        dims = [input_dim, *hidden, 1]
        layers: list[nn.Module] = []
        for idx, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(d_in, d_out))
            if idx < len(dims) - 2:
                layers.extend((nn.SiLU(), nn.Dropout(dropout)))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x).squeeze(-1)


class EdgeGatedLayer(nn.Module):
    """Dense edge-conditioned message passing, bond-only gate, GRU residual."""
    def __init__(self, hidden_dim: int = 128, bond_dim: int = 8, dropout: float = 0.1):
        super().__init__()
        self.gate_first = nn.Linear(bond_dim, hidden_dim)
        self.gate_second = nn.Linear(hidden_dim, hidden_dim)
        self.message = nn.Linear(hidden_dim, hidden_dim)
        self.cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: Tensor, bonds: Tensor, adjacency: Tensor, atom_mask: Tensor) -> Tensor:
        gate = torch.sigmoid(self.gate_second(F.silu(self.gate_first(bonds))))
        neighbor_messages = self.message(h).unsqueeze(1)  # [B,1,neighbor,hidden]
        aggregated = (gate * neighbor_messages * adjacency.unsqueeze(-1)).sum(dim=2)
        b, n, d = h.shape
        delta = self.cell(aggregated.reshape(b * n, d), h.reshape(b * n, d)).reshape(b, n, d)
        return (h + self.dropout(delta)) * atom_mask.unsqueeze(-1)


class GraphMPNN(nn.Module):
    """Manuscript-width edge-gated GraphMPNN (36 atom, 8 bond, 128 hidden, 4 layers).

    `site_readout=False` for basic pKa; site flag may still be present in the 36
    atom features. `site_index=-1` represents no matched site.
    """
    def __init__(self, *, site_readout: bool, atom_dim: int = 36,
                 bond_dim: int = 8, hidden_dim: int = 128, layers: int = 4,
                 dropout: float = 0.1, head_dropout: float = 0.2):
        super().__init__()
        self.site_readout = site_readout
        self.atom_projection = nn.Linear(atom_dim, hidden_dim)
        self.layers = nn.ModuleList([EdgeGatedLayer(hidden_dim, bond_dim, dropout) for _ in range(layers)])
        pooled_dim = hidden_dim * (3 if site_readout else 2)
        self.norm = nn.LayerNorm(pooled_dim)
        self.head = nn.Sequential(nn.Linear(pooled_dim, hidden_dim), nn.SiLU(),
                                  nn.Dropout(head_dropout), nn.Linear(hidden_dim, 1))

    def forward(self, atoms: Tensor, bonds: Tensor, adjacency: Tensor,
                atom_mask: Tensor, site_index: Tensor) -> Tensor:
        if atoms.ndim != 3 or bonds.ndim != 4:
            raise ValueError("Expected atoms [B,N,36] and bonds [B,N,N,8]")
        if torch.any(adjacency.diagonal(dim1=1,dim2=2)):
            raise ValueError("Graph adjacency must have zero diagonal")
        h = F.silu(self.atom_projection(atoms)) * atom_mask.unsqueeze(-1)
        for layer in self.layers:
            h = layer(h, bonds, adjacency, atom_mask)
        count = atom_mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled_mean = h.sum(dim=1) / count
        pooled_max = h.masked_fill(~atom_mask.unsqueeze(-1), float("-inf")).amax(dim=1)
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))
        pieces = []
        if self.site_readout:
            idx = site_index.clamp_min(0).clamp_max(h.shape[1]-1)
            selected = h[torch.arange(h.shape[0], device=h.device), idx]
            pieces.append(selected * (site_index >= 0).unsqueeze(-1))
        pieces.extend((pooled_mean, pooled_max))
        return self.head(self.norm(torch.cat(pieces, dim=-1))).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())
