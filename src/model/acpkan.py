import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.attn(x)
        return x * weights


class ChebPolyLayer(nn.Module):
    def __init__(self, in_dim, out_dim, poly_degree=2):
        super().__init__()
        self.poly_degree = poly_degree
        self.weight = nn.Parameter(torch.Tensor(out_dim, in_dim * (poly_degree + 1)))
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        x_t = torch.tanh(x)
        T_list = [torch.ones_like(x_t), x_t]
        for n in range(2, self.poly_degree + 1):
            Tn = 2 * x_t * T_list[-1] - T_list[-2]
            T_list.append(Tn)
        T_stack = torch.stack(T_list, dim=2)
        T_flat = T_stack.view(x.shape[0], -1)
        return T_flat @ self.weight.t() + self.bias


class ACPKAN_Deterministic(nn.Module):
    def __init__(self, input_dim, hidden_dim=6, output_dim=1, poly_degree=2, dropout_p=0.1):
        super().__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.cheb = ChebPolyLayer(hidden_dim, hidden_dim, poly_degree)
        self.attn = ChannelAttention(hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.out_layer = nn.Linear(hidden_dim, output_dim)
        self.regularizer = None  # Placeholder for regularization if needed

    def forward(self, x):
        x = self.linear_in(x)
        x = x + self.attn(self.cheb(x))
        x = self.dropout(x)
        out = self.out_layer(x)
        return out


if __name__ == "__main__":
    dummy_input = torch.randn(10, 7)
    model = ACPKAN_Deterministic(input_dim=7, hidden_dim=6, output_dim=1, poly_degree=2)
    predictions = model(dummy_input)
    print("Output Shape:", predictions.shape)
