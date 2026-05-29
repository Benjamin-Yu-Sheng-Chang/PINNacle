import torch
import torch.nn as nn


class ReLUk(nn.Module):
    """ReLUk activation function: ReLU(x)^k"""
    
    def __init__(self, k=1):
        super().__init__()
        self.k = k
    
    def forward(self, x):
        return torch.relu(x) ** self.k


class MLPReLUk(nn.Module):
    """MLP with ReLUk activation function.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        depth: Number of layers (including input and output layers)
        k: Exponent for ReLUk activation (default: 1)
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, depth, k=1):
        super().__init__()
        
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(ReLUk(k))
        
        # Hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(ReLUk(k))
        
        # Output layer (no activation)
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self.regularizer = None  # Placeholder for regularization if needed
    
    def forward(self, x):
        return self.network(x)


class StandardMLP(nn.Module):
    """Standard MLP with Tanh activation (original implementation).
    
    Args:
        layer_sizes: List of layer sizes [input_dim, hidden_dim, ..., output_dim]
    """
    
    def __init__(self, layer_sizes):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            # Add activation after every hidden linear (not after final layer)
            if i < len(layer_sizes) - 2:
                layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        self.regularizer = None  # Placeholder for regularization if needed
    
    def forward(self, x):
        return self.network(x)
