import torch
import torch.nn as nn
import torch.nn.functional as F
from noisy_layer import NoisyLinear

class Network(nn.Module):
    """Network class for Categorical DQN."""
    def convs(self, x):
        x = self.feature_layer(x)
        if self._to_linear is None:
            self._to_linear = x[0].numel() 
        return x.view(-1, self._to_linear)
    
    def _flatten_dims(self, dims: tuple[int, int, int]) -> int:
        """Flatten dimensions."""
        return dims[0] * dims[1] * dims[2]
    
    def __init__(
        self, 
        in_dim: tuple[int, int, int], 
        out_dim: int, 
        atom_size: int, 
        support: torch.Tensor
    ):
        """Initialization."""
        super(Network, self).__init__()
        
        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Conv2d(in_dim[0], 8, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Determine the output size by doing a forward pass with a mock input
        x = torch.randn(1, in_dim[0], in_dim[1], in_dim[2])
        self._to_linear = None
        self.convs(x) 
        print(self._to_linear)
        self.fc = nn.Linear(self._to_linear, 128) 
        self.advantage_hidden_layer = NoisyLinear(128, 128)
        self.advantage_layer = NoisyLinear(128, out_dim * atom_size)

        # set value layer
        self.value_hidden_layer = NoisyLinear(128, 128)
        self.value_layer = NoisyLinear(128, atom_size)


    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        
        return q
    
    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        feature = self.convs(x)
        feature = F.relu(self.fc(feature))
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))
        
        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size
        )
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()