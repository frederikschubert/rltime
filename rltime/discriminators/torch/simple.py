import torch
from .torch_disciminator import TorchDiscriminator

class Simple(TorchDiscriminator):

    def predict(self, x, timesteps):
        # Perform a forward pass on the model
        res = self.model(x, timesteps)
        # Apply the actions linear layer to the output
        output = torch.sigmoid(self.out_layer(res['output']))
        return output