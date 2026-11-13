"""``MLPclassifier`` of the TwiBot-22 benchmark RoBERTa baseline (spec section 4).

    p1 = Linear(768, 384)(tweet)            no activation
    p2 = Linear(768, 384)(des)              no activation
    x  = cat(p1, p2)                        [B, 768]
    x  = LeakyReLU(Linear(768, 128)(x))     the only nonlinearity ("linear_relu_tweet")
    x  = Dropout(p)(x)
    logits = Linear(128, 2)(x)

Inputs arrive as one fp16 block ``[B, 1536] = [tweet | des]`` and are cast to fp32 here. The
unused modules of the original class are not instantiated (spec 4). Init: PyTorch defaults.

ExtScore ``[z, m]`` (spec 8), only when ``extscore_width > 0``:
- ``fuse`` (default): concatenated to ``cat(p1, p2)``, so ``Linear(770, 128)``;
- ``late``: concatenated after the dropout, so ``Linear(130, 2)``;
- ``branch``: ``p3 = Linear(2, k)([z, m])`` concatenated at step 3, so ``Linear(768 + k, 128)``.
With ``extscore_width == 0`` the module, its parameter shapes and creation order are the baseline's.
"""
from __future__ import annotations

import torch
from torch import nn

EMB = 768
EXTSCORE_INJECT = ("fuse", "late", "branch")


class RobertaMLP(nn.Module):
    def __init__(self, dropout: float, *, emb: int = EMB, proj: int = 384, hidden: int = 128,
                 extscore_width: int = 0, extscore_inject: str = "fuse", branch_dim: int = 8):
        super().__init__()
        if extscore_inject not in EXTSCORE_INJECT:
            raise ValueError(f"extscore_inject must be one of {EXTSCORE_INJECT}")
        self.emb, self.extscore_width = emb, extscore_width
        self.extscore_inject = extscore_inject if extscore_width else None
        self.tweet_proj = nn.Linear(emb, proj)
        self.des_proj = nn.Linear(emb, proj)
        self.extscore_proj = nn.Linear(extscore_width, branch_dim) if self.extscore_inject == "branch" else None
        fuse_in = 2 * proj + {None: 0, "fuse": extscore_width, "late": 0, "branch": branch_dim}[self.extscore_inject]
        self.linear_relu_tweet = nn.Sequential(nn.Linear(fuse_in, hidden), nn.LeakyReLU())
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden + (extscore_width if self.extscore_inject == "late" else 0), 2)

    def forward(self, x: torch.Tensor, extscore: torch.Tensor | None = None) -> torch.Tensor:
        x = x.float()
        h = [self.tweet_proj(x[:, : self.emb]), self.des_proj(x[:, self.emb:])]
        if self.extscore_inject == "fuse":
            h.append(extscore)
        elif self.extscore_inject == "branch":
            h.append(self.extscore_proj(extscore))
        h = self.dropout(self.linear_relu_tweet(torch.cat(h, 1)))
        if self.extscore_inject == "late":
            h = torch.cat([h, extscore], 1)
        return self.classifier(h)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
