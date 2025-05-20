import torch
import torch.nn.init as init
from torch import nn


class CNNStencilModel(nn.Module):
    def __init__(self, depth=6, hidden=15, act=nn.ReLU, last_act=None, clip=1.0, dtype=torch.float64):
        """
        Initialize the CNN model to be used as a numerical scheme.

        Args:
            model_type: one of 'density', 'density_delta', 'flow', 'speed'.
            depth: number of hidden CNN layers.
            hidden: number of channels in each hidden layer.
            act: activation function to use between layers.
        """
        super().__init__()

        # create the model
        layers = []
        layers.append(nn.Conv1d(in_channels=1, out_channels=hidden, kernel_size=2, padding=0, dtype=dtype))
        layers.append(act())
        for _ in range(depth - 1):
            layers.append(nn.Conv1d(in_channels=hidden, out_channels=hidden, kernel_size=1, padding=0, dtype=dtype))
            layers.append(act())
        layers.append(nn.Conv1d(in_channels=hidden, out_channels=1, kernel_size=1, padding=0, dtype=dtype))
        if last_act:
            layers.append(last_act())
        self.model = nn.Sequential(*layers)
        self._init_weights(act)

        self.clip = clip

    def _init_weights(self, act):
        """Initialize weights based on the activation function."""
        for m in self.model.modules():
            if isinstance(m, nn.Conv1d):
                if act == nn.ReLU:
                    init.kaiming_normal_(m.weight, nonlinearity="relu")
                elif act == nn.ELU:
                    init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                elif act == nn.Tanh:
                    init.xavier_normal_(m.weight)
                else:
                    print(f"Warning: Activation function {act} not supported in neural network weights initialization.")
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, rho_BX, *args):
        """
        Forward pass of the model.

        Args:
            rho_BX: torch.tensor of shape (B, X), i.e. densities at time t.

        Returns:
            flux_BX: torch.tensor of shape (B, X-1), i.e. numerical flux between densities from t to t+1.
        """
        # add a channel dimension C=1 for the convolutions to work properly
        if self.clip is None:
            return self.model(rho_BX.unsqueeze(1))[:, 0, :]
        else:
            return torch.clamp(self.model(rho_BX.unsqueeze(1)), 0.0, self.clip)[:, 0, :]

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def load_checkpoint(self, checkpoint_path, device="cpu"):
        state_dict = torch.load(checkpoint_path, weights_only=True, map_location=torch.device(device))

        if "model.17.bias" in state_dict:  # drone
            for x in ["weight", "bias"]:
                state_dict[f"model.0.{x}"] = state_dict[f"model.0.{x}"]
                state_dict[f"model.2.{x}"] = state_dict[f"model.2.{x}"]
                state_dict[f"model.4.{x}"] = state_dict[f"model.5.{x}"]
                state_dict[f"model.6.{x}"] = state_dict[f"model.8.{x}"]
                state_dict[f"model.8.{x}"] = state_dict[f"model.11.{x}"]
                state_dict[f"model.10.{x}"] = state_dict[f"model.14.{x}"]
                state_dict[f"model.12.{x}"] = state_dict[f"model.17.{x}"]

                del state_dict[f"model.5.{x}"]
                del state_dict[f"model.11.{x}"]
                del state_dict[f"model.14.{x}"]
                del state_dict[f"model.17.{x}"]

        if "flowModel.stencil" in state_dict:
            # to load models from Victor's code
            state_dict_map = {
                "conv.Conv1.weight": "model.0.weight",
                "conv.Conv1.bias": "model.0.bias",
                "conv.fc_0.weight": "model.2.weight",
                "conv.fc_0.bias": "model.2.bias",
                "conv.fc_1.weight": "model.4.weight",
                "conv.fc_1.bias": "model.4.bias",
                "conv.fc_2.weight": "model.6.weight",
                "conv.fc_2.bias": "model.6.bias",
                "conv.fc_3.weight": "model.8.weight",
                "conv.fc_3.bias": "model.8.bias",
                "conv.fc_4.weight": "model.10.weight",
                "conv.fc_4.bias": "model.10.bias",
                "conv.fc_end.weight": "model.12.weight",
                "conv.fc_end.bias": "model.12.bias",
            }
            for k in ["stencil", "ReLU", "Tanh", "ELU"]:
                full_k = "flowModel." + k
                if full_k in state_dict:
                    del state_dict[full_k]
            state_dict = {
                state_dict_map[k[len("flowModel.") :]]: v
                for k, v in state_dict.items()
                if "stencil" not in k and "ELU" not in k and "Tanh" not in k
            }

        self.load_state_dict(state_dict)

    @property
    def device(self):
        devices = set(p.device for p in self.parameters())
        if len(devices) > 1:
            raise ValueError("Model has multiple devices")
        return devices.pop()

    @property
    def dtype(self):
        dtypes = set(p.dtype for p in self.parameters())
        if len(dtypes) > 1:
            raise ValueError("Model has multiple dtypes")
        return dtypes.pop()
