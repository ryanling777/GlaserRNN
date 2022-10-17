from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from .utils import glorot_gauss_tensor

@dataclass
class ConnectionConfig:
    source_name: str
    target_name: str
    rank: Union[int, str] = 'full'
    p_conn: float = 1.
    train_weights: bool = True
    mask: torch.Tensor = None # for masking input lines such as go cue

    @property
    def weight_name(self):
        return f'W_{self.source_name}_to_{self.target_name}'


class Connection(nn.Module):
    def __init__(self, config, N_from, N_to):
        super().__init__()

        self.config = config
        self.source_name = config.source_name
        self.target_name = config.target_name

        if self.config.rank is None:
            self.rank = 'full'
        else:
            self.rank = self.config.rank

        self.connect(N_from, N_to)

    def connect(self, N_from, N_to):
        sparse_mask = torch.rand(N_to, N_from) < self.config.p_conn
        self.register_buffer(f'sparse_mask', sparse_mask)

        self.W = nn.Parameter(glorot_gauss_tensor(connectivity=sparse_mask), requires_grad = self.config.train_weights)

        if self.rank != 'full':
            assert isinstance(self.rank, int)
            import geotorch
            geotorch.low_rank(self, 'W', self.rank)

    @property
    def effective_W(self):
        if self.config.mask is None:
            return self.W * self.sparse_mask
        else:
            return self.W * self.sparse_mask * self.config.mask
        
    #def __getattr__(self, name):
    #    return getattr(self.config, name)
