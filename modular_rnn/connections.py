from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn

from .utils import glorot_gauss_tensor

@dataclass
class ConnectionConfig:
    source_name: str
    target_name: str
    rank: Union[int, str] = 'full'
    p_conn: float = 1.
    norm: Union[float, None] = None
    #orthogonal: bool = False
    train_weights_direction: bool = True
    mask: torch.Tensor = None # for masking input lines such as go cue

    @property
    def weight_name(self):
        return f'W_{self.source_name}_to_{self.target_name}'


class Connection(nn.Module):
    def __init__(self, config, N_from, N_to):
        super().__init__()

        self.source_name = config.source_name
        self.target_name = config.target_name

        assert (config.rank == 'full') or isinstance(config.rank, int)
        self.rank = config.rank

        assert (config.norm is None) or isinstance(config.norm, float)
        self.norm = config.norm

        #self.orthogonal = config.orthogonal

        self.train_weights_direction = config.train_weights_direction

        self.p_conn = config.p_conn
        self.mask = config.mask

        self.connect(N_from, N_to)

    def connect(self, N_from: int, N_to: int) -> None:
        sparse_mask = torch.rand(N_to, N_from) < self.p_conn
        self.register_buffer(f'sparse_mask', sparse_mask)

        self.W = nn.Parameter(glorot_gauss_tensor(connectivity=sparse_mask),
                              requires_grad = self.train_weights_direction)

        if self.rank != 'full':
            assert isinstance(self.rank, int)
            import geotorch
            geotorch.low_rank(self, 'W', self.rank)

        if self.norm is not None:
            assert isinstance(self.norm, float)
            import geotorch
            geotorch.sphere(self, 'W', radius = self.norm)

        # TODO handle orthogonal and norm together
        # right now orthogonal makes the columns have unit norm
        # if orthogonal and norm are both given, have an extra parameter that handles the lengths
        # don't set sphere, only orthogonal, and multiply by that extra parameter
        #if self.orthogonal:
        #    import geotorch
        #    geotorch.orthogonal(self, 'W')


    @property
    def effective_W(self) -> torch.Tensor:
        if self.mask is None:
            return self.W * self.sparse_mask
        else:
            return self.W * self.sparse_mask * self.mask
        
    #def __getattr__(self, name):
    #    return getattr(self.config, name)

    def __repr__(self):
        return str(self.__dict__)
