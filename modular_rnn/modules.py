from typing import Union

import torch
import torch.nn as nn

from .utils import glorot_gauss_tensor


class RNNModule(nn.Module):
    def __init__(self,
                 name: str,
                 n_neurons: int,
                 alpha: float, 
                 nonlin: callable,
                 p_rec: float = 1.,
                 rec_rank: Union[int, None] = None,
                 train_recurrent_weights: bool = True, # TODO use this
                 dynamics_noise: Union[float, None] = None,
                 bias: bool = True,
                 use_constant_init_state: bool = False):
        super().__init__()

        self.name = name
        self.n_neurons = n_neurons
        self.alpha = alpha
        self.nonlin = nonlin
        self.p_rec = p_rec
        self.nonlin_fn = nonlin
        self.bias = bias
        self.use_constant_init_state = use_constant_init_state
        self.train_recurrent_weights = train_recurrent_weights


        if dynamics_noise is None:
            self.noisy = False
            self.noise_amp = 0
        else:
            self.noisy = True
            self.noise_amp = dynamics_noise

        self.glorot_gauss_init()

        if rec_rank is not None:
            import geotorch
            geotorch.low_rank(self, 'W_rec', rec_rank)
            self.rec_rank = rec_rank
        else:
            self.rec_rank = 'full'

        # for potentially initializing to the same state in every trial
        init_x = self.sample_random_hidden_state_vector()
        self.register_buffer('init_x', init_x)


    def sample_random_hidden_state_vector(self) -> torch.Tensor:
        return .1 + .01 * torch.randn(self.n_neurons)


    def init_hidden(self) -> None:
        if self.use_constant_init_state:
            init_state = torch.tile(self.init_x, (1, self.batch_size, 1))
        else:
            init_state = torch.tile(self.sample_random_hidden_state_vector(), (1, self.batch_size, 1))

        # might need self.device
        self.hidden_states = [init_state]
        self.rates = [self.nonlin(init_state)]


    def f_step(self) -> None:

        x, r = self.hidden_states[-1], self.rates[-1]

        x = x + self.alpha * (-x + r @ (self.rec_mask * self.W_rec).T
                                 + self.inputs_at_current_time
                                 + self.bias)

        if self.noisy:
            x += self.alpha * self.noise_amp * torch.randn(1, self.batch_size, self.n_neurons).to(self.device)

        r = self.nonlin_fn(x)

        self.hidden_states.append(x)
        self.rates.append(r)


    def get_hidden_states_tensor(self) -> torch.Tensor:
        return torch.stack(self.hidden_states).squeeze()


    @property
    def rates_tensor(self) -> torch.Tensor:
        return torch.stack(self.rates).squeeze()


    def glorot_gauss_init(self) -> None:
        rec_mask = torch.rand(self.n_neurons, self.n_neurons) < self.p_rec
        self.W_rec = nn.Parameter(glorot_gauss_tensor(connectivity=rec_mask))
        self.register_buffer('rec_mask', rec_mask)

        if self.bias:
            self.bias = nn.Parameter(glorot_gauss_tensor((1, self.n_neurons)))
        else:
            self.bias = nn.Parameter(torch.zeros((1, self.n_neurons)))
            self.bias.requires_grad = False

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def norm_rec(self):
        return (self.rec_mask * self.W_rec).norm()

    def __repr__(self):
        return str(self.__dict__)


class ModelOutput:
    def __init__(self, name: str, dim: int):
        self.name = name
        self.dim = dim
        self.values = None
    
    def reset(self, batch_size: int):
        self.values = [torch.zeros(1, batch_size, self.dim)]
        
    def as_tensor(self):
        return torch.stack(self.values).squeeze()

    #def __getitem__(self, item):
    #    return self.as_tensor()[item]

    def __getattr__(self, name):
        return getattr(self.as_tensor(), name)
