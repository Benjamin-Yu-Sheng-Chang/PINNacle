import torch
from kan import KAN  # Ensure pykan is installed: pip install pykan

class DNN_PIKAN(torch.nn.Module):
    def __init__(self, hidden_layers, x_dim=1, u_dim=1, grid_size=5, spline_order=3):
        """
        Args:
            hidden_layers (list): List of integers representing hidden layer widths.
            x_dim (int): Number of input variables.
            u_dim (int): Number of output state variables.
            grid_size (int): Number of grid intervals for B-splines.
            spline_order (int): Order of B-splines.
        """
        super(DNN_PIKAN, self).__init__()
        
        self.x_dim = x_dim
        self.u_dim = u_dim
        
        # Construct the complete layer layout for KAN
        # e.g., if x_dim=2, hidden_layers=[100, 100], u_dim=1 -> [2, 100, 100, 1]
        self.layer_sizes = [self.x_dim] + hidden_layers + [self.u_dim]

        self.regularizer = None  # Placeholder for regularization if needed
        
        # Instantiate the Kolmogorov-Arnold Network
        self.kan = KAN(
            width=self.layer_sizes,
            grid=grid_size,
            k=spline_order
        )

    def forward(self, x):
        # Explicitly match type casting like the baseline script does
        return self.kan(x.float())