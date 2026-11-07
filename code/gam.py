"""Data, baselines, G-NAMRFF re-implementation and the proposed hop-additive GAM."""

import math
import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.datasets import Actor, HeterophilousGraphDataset, Planetoid, WebKB
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import get_laplacian, to_undirected

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
AUC_SETS = {"minesweeper", "tolokers", "questions"}


def load(name):
    """Return (data, masks) with masks = (train, val, test), each N x S boolean."""
    low = name.lower()
    if low in ("cora", "citeseer", "pubmed"):
        g = Planetoid(
            os.path.join(DATA, "planetoid"), name, transform=NormalizeFeatures()
        )[0]
    elif low in ("cornell", "texas", "wisconsin"):
        g = WebKB(os.path.join(DATA, "webkb"), name)[0]
    elif low == "actor":
        g = Actor(os.path.join(DATA, "actor"))[0]
    else:
        g = HeterophilousGraphDataset(os.path.join(DATA, "hetero"), name)[0]
    g.edge_index = to_undirected(g.edge_index, num_nodes=g.num_nodes)
    masks = [getattr(g, f"{s}_mask") for s in ("train", "val", "test")]
    masks = [m.unsqueeze(1) if m.dim() == 1 else m for m in masks]
    return g, masks


def metric(name, logits, y):
    if name.lower() in AUC_SETS:
        from sklearn.metrics import roc_auc_score

        return float(
            roc_auc_score(y.cpu().numpy(), logits[:, 1].detach().cpu().numpy())
        )
    return float((logits.argmax(1) == y).float().mean())


# --------------------------------------------------------------------- baselines
class MLP(nn.Module):
    def __init__(self, d, c, hidden=64, drop=0.5, **_):
        super().__init__()
        self.a, self.b, self.drop = nn.Linear(d, hidden), nn.Linear(hidden, c), drop

    def forward(self, x, ei):
        x = F.dropout(x, self.drop, self.training)
        return self.b(F.dropout(F.relu(self.a(x)), self.drop, self.training))


class Conv2(nn.Module):
    def __init__(self, d, c, hidden=64, drop=0.5, kind="gcn", heads=4, **_):
        super().__init__()
        if kind == "gcn":
            self.a, self.b = GCNConv(d, hidden), GCNConv(hidden, c)
        elif kind == "sage":
            self.a, self.b = SAGEConv(d, hidden), SAGEConv(hidden, c)
        else:
            self.a = GATConv(d, hidden // heads, heads=heads, dropout=drop)
            self.b = GATConv(hidden, c, heads=1, dropout=drop)
        self.drop = drop

    def forward(self, x, ei):
        x = F.dropout(x, self.drop, self.training)
        return self.b(F.dropout(F.relu(self.a(x, ei)), self.drop, self.training), ei)


def laplacian_powers(g, R):
    """[L^0 X, ..., L^(R-1) X]. Matches the authors' code, whose R counts terms, not order."""
    ei, ew = get_laplacian(g.edge_index, normalization="sym", num_nodes=g.num_nodes)
    L = (
        torch.sparse_coo_tensor(ei, ew, (g.num_nodes, g.num_nodes))
        .coalesce()
        .to(g.x.device)
    )
    out = [g.x]
    for _ in range(R - 1):
        out.append(torch.sparse.mm(L, out[-1]))
    return torch.stack(out)


class LinFilter(nn.Module):
    """Control: G-NAMRFF's filter followed by a plain linear classifier."""

    def __init__(self, P, c, **_):
        super().__init__()
        self.P, self.alpha = P, nn.Parameter(torch.randn(P.size(0)))
        self.head = nn.Linear(P.size(2), c)

    def forward(self, x, ei):
        return self.head((self.alpha.view(-1, 1, 1) * self.P).sum(0))


class GNAMRFF(nn.Module):
    """Function-identical to the authors' NAMGraphRFFModel, with sparse filtering and chunking."""

    def __init__(self, P, c, M=100, theta=1.0, chunk=64, **_):
        super().__init__()
        self.P, self.chunk, self.M = P, chunk, M
        self.alphas = nn.Parameter(torch.randn(P.size(0)))
        self.register_buffer("a", torch.randn(M, 1) / theta)
        self.register_buffer("b", 2 * math.pi * torch.rand(M))
        self.w = nn.Parameter(torch.randn(P.size(2), M + 1, c))

    def filtered(self):
        return (self.alphas.view(-1, 1, 1) * self.P).sum(0)

    def _part(self, xt, w):
        phi = torch.cos(xt.unsqueeze(-1) * self.a.T + self.b) * math.sqrt(2.0 / self.M)
        return torch.einsum("nkm,kmc->nc", phi, w[:, :-1]) + w[:, -1].sum(0)

    def forward(self, x, ei):
        xt, out = self.filtered(), 0
        for s in range(0, xt.size(1), self.chunk):
            xs, ws = xt[:, s : s + self.chunk], self.w[s : s + self.chunk]
            out = out + (
                checkpoint(self._part, xs, ws, use_reentrant=False)
                if self.training
                else self._part(xs, ws)
            )
        return out


# --------------------------------------------------------------------- proposed
def hop_channels(g, H):
    """[X, mean over exact-distance-1 neighbours of X, ..., distance H]. Each channel has a true hop meaning."""
    n = g.num_nodes
    ei = g.edge_index.cpu().numpy()
    A = sp.csr_matrix(
        (np.ones(ei.shape[1], dtype=np.float32), (ei[0], ei[1])), shape=(n, n)
    )
    A.setdiag(0)
    A.eliminate_zeros()
    A.data[:] = 1
    seen = (
        (A + sp.eye(n, format="csr", dtype=np.float32))
        .astype(bool)
        .astype(np.float32)
        .tocsr()
    )
    frontier, X = A, g.x.cpu().numpy()
    out = [X]
    for h in range(1, H + 1):
        deg = np.asarray(frontier.sum(1)).ravel()
        norm = sp.diags(
            np.where(deg > 0, 1.0 / np.maximum(deg, 1), 0.0).astype(np.float32)
        )
        out.append(np.asarray((norm @ frontier) @ X, dtype=np.float32))
        if h == H:
            break
        nxt = (frontier @ A).astype(bool).astype(np.float32).tocsr()
        nxt = (nxt - nxt.multiply(seen)).tocsr()
        nxt.eliminate_zeros()
        seen = (seen + nxt).astype(bool).astype(np.float32).tocsr()
        frontier = nxt
    return out


def mercer_basis(sample, theta, K, grid=512):
    """Deterministic top-K centred eigenfunctions of the RBF kernel under the empirical input law."""
    q = torch.quantile(sample, torch.linspace(0, 1, grid, device=sample.device))
    q = torch.unique(q)
    if q.numel() == 1:
        q = torch.stack([q[0] - 1, q[0] + 1])
    Kgg = torch.exp(-((q[:, None] - q[None, :]) ** 2) / (2 * theta**2)).double()
    m = q.numel()
    C = torch.eye(m, device=q.device).double() - 1.0 / m
    lam, V = torch.linalg.eigh(C @ Kgg @ C)
    keep = lam.argsort(descending=True)[:K]
    keep = keep[lam[keep] > 1e-9 * lam.max()]
    lam, V = lam[keep], V[:, keep]
    mean_col = Kgg.mean(1, keepdim=True)
    mean_all = Kgg.mean()

    def psi(z):
        Kzg = torch.exp(-((z.unsqueeze(-1) - q) ** 2) / (2 * theta**2)).double()
        Kc = Kzg - Kzg.mean(-1, keepdim=True) - mean_col.squeeze() + mean_all
        return ((Kc @ V) * math.sqrt(m) / lam).float()

    return psi, (lam / m).float()


class HopGAM(nn.Module):
    """logit_ic = b_c + sum_{k,h,j} W[k,h,j,c] psi_j(z_{ikh}). Linear in W, hence convex with ridge."""

    def __init__(
        self, g, c, H=2, K=4, theta=1.0, clip=5.0, device="cuda", max_nnz=None
    ):
        super().__init__()
        chans = hop_channels(g, H)
        Z = torch.from_numpy(np.stack(chans, 2)).to(device)  # N x D x (H+1)
        self.mean, self.std = Z.mean(0, keepdim=True), Z.std(0, keepdim=True).clamp_min(
            1e-8
        )
        Z = ((Z - self.mean) / self.std).clamp(-clip, clip)
        idx = torch.randint(
            0,
            Z.numel(),
            (min(2_000_000, Z.numel()),),
            generator=torch.Generator().manual_seed(0),
        )
        self.psi, self.lam = mercer_basis(Z.reshape(-1)[idx.to(device)], theta, K)
        K = self.lam.numel()
        # psi is smooth, so tabulate it finely and interpolate instead of evaluating the kernel per entry.
        tab_x = torch.linspace(-clip, clip, 8193, device=Z.device)
        tab = torch.cat([self.psi(tab_x[s : s + 1024]) for s in range(0, 8193, 1024)])
        pos = (Z + clip) / (2 * clip) * 8192
        lo = pos.floor().long().clamp(0, 8191)
        frac = (pos - lo).unsqueeze(-1)
        Psi = tab[lo] * (1 - frac) + tab[lo + 1] * frac
        self.register_buffer("Psi", Psi)  # N x D x (H+1) x K
        self.d, self.h, self.k, self.c = Z.size(1), H + 1, K, c
        self.W = nn.Parameter(torch.zeros(self.d, self.h, self.k, c))
        self.bias = nn.Parameter(torch.zeros(c))

    def forward(self, x=None, ei=None):
        return torch.einsum("ndhj,dhjc->nc", self.Psi, self.W) + self.bias

    def contributions(self):
        """Exact tensor N x D x (H+1) x C summing to logits - bias."""
        return torch.einsum("ndhj,dhjc->ndhc", self.Psi, self.W)

    def penalty(self):
        return (self.W**2 * (self.lam.view(1, 1, -1, 1) ** -1 * self.lam.max())).sum()
