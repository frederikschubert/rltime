import torch
import torch.nn as nn

from ..disciminator import Discriminator
from rltime.models.torch.utils import make_tensor
from rltime.models.torch.utils import linear
from rltime.policies.torch.torch_policy import torch_get_raw, torch_load_cpu

class TorchDiscriminator(nn.Module, Discriminator):

    def __init__(self, observation_space, model_config):
        super().__init__()
        # Create the model
        self.model = self._create_model_from_config(
            model_config, observation_space)
        self.is_cuda = self.model.is_cuda
        self.out_layer = linear(self.model.out_size, 1)

    @classmethod
    def create(cls, *args, cuda="auto", **kwargs):
        """Creates a torch discriminator with the given args+kwargs

        Args:
            cuda: configures the cuda-placement of the discriminator. 'auto' means to
                use the default cuda device if available otherwise the cpu.
                Can also be True/False or a specific cuda device for example
                "cuda:1"
        """
        discriminator = cls(*args, **kwargs)
        if cuda == "auto":
            cuda = torch.cuda.is_available()
        if cuda:
            discriminator = discriminator.to(torch.device("cuda" if cuda is True else cuda))

        return discriminator
    
    def make_input_state(self, inp, initials):
        return self.model.make_input_state(inp, initials)

    def is_recurrent(self):
        return self.model.is_recurrent()

    def get_creator(self, cuda=False):
        # Returns a lambda that generates a copy of this model
        # NOTE: cuda option not supported ATM
        data = torch_get_raw(self)
        return lambda: torch_load_cpu(data)

    def get_state(self):
        return torch_get_raw(self.state_dict())

    def load_state(self, state):
        self.load_state_dict(torch_load_cpu(state))

    def get_grad_norm(self):
        """Calculates the global grad-norm (L2-variant) of the model
        parameters"""
        total_norm = 0
        for p in self.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def make_tensor(self, x, non_blocking=False):
        """Makes a tensor out of the given input, on the policies device"""
        return make_tensor(x, device="cpu" if not self.is_cuda() else "cuda",
                           non_blocking=non_blocking)