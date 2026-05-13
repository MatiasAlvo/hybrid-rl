import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Q(s, a) network for learned-critic pathwise training.
    Mirrors ValueNetwork architecture/initialization and takes
    concatenated (observation, action) as input.
    """

    def __init__(self, config, device="cpu"):
        super().__init__()
        self.device = device
        self.config = config
        self.trainable = True

        input_size = config.get("input_size", None)
        self.obs_size = None if input_size is None else int(input_size)
        self.action_size = int(config["action_size"])

        self.backbone = self._build_backbone()
        self.head = self._build_head()
        self.to(device)

    def lazy_layer_init(self, layer, std=2**0.5, bias_const=0.0):
        def init_hook(module, input):
            if not hasattr(module, "initialized"):
                torch.nn.init.orthogonal_(module.weight, std)
                torch.nn.init.constant_(module.bias, bias_const)
                module.initialized = True

        layer.register_forward_pre_hook(init_hook)
        return layer

    def layer_init(self, layer, std=2**0.5, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def _get_activation(self, name):
        return {
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "tanh": nn.Tanh(),
            "linear": nn.Identity(),
        }[name.lower()]

    def _build_backbone(self):
        layers = []
        hidden_layers = self.config["hidden_layers"]
        dropout = self.config.get("dropout", 0.0)
        use_batch_norm = self.config.get("batch_norm", False)
        activation_name = self.config["activation"]

        if self.obs_size is None:
            first_layer = self.lazy_layer_init(nn.LazyLinear(hidden_layers[0]), std=2**0.5)
        else:
            input_size = self.obs_size + self.action_size
            first_layer = self.layer_init(nn.Linear(input_size, hidden_layers[0]), std=2**0.5)
        layers.append(first_layer)
        layers.append(self._get_activation(activation_name))
        layers.append(nn.Dropout(dropout))

        for i in range(1, len(hidden_layers)):
            prev_size = hidden_layers[i - 1]
            size = hidden_layers[i]
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(prev_size))
            linear_layer = self.layer_init(nn.Linear(prev_size, size), std=2**0.5)
            layers.extend(
                [
                    linear_layer,
                    self._get_activation(activation_name),
                    nn.Dropout(dropout),
                ]
            )

        return nn.Sequential(*layers)

    def _build_head(self):
        return self.layer_init(nn.Linear(self.config["hidden_layers"][-1], 1), std=1.0)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        features = self.backbone(x)
        return self.head(features).squeeze(-1)
