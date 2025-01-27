import torch
import torch.nn as nn

class ValueNetwork(nn.Module):
    """Value network for actor-critic architectures"""
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.device = device
        self.config = config
        self.trainable = True
        
        # Build network components
        self.backbone = self._build_backbone()
        self.head = self._build_head()
        self.to(device)
        
        # observation_keys will be set by HybridAgent
        self.observation_keys = None
    
    def lazy_layer_init(self, layer, std=2**0.5, bias_const=0.0):
        def init_hook(module, input):
            if not hasattr(module, 'initialized'):
                torch.nn.init.orthogonal_(module.weight, std)
                torch.nn.init.constant_(module.bias, bias_const)
                module.initialized = True
        
        layer.register_forward_pre_hook(init_hook)
        return layer
    
    def layer_init(self, layer, std=2**0.5, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    
    def _build_backbone(self):
        layers = []
        
        # Use LazyLinear for first layer with lazy initialization
        first_layer = self.lazy_layer_init(
            nn.LazyLinear(self.config['hidden_layers'][0]),
            std=2 ** 0.5  # Using scalar instead of torch.sqrt(2)
        )
        layers.append(first_layer)
        layers.append(self._get_activation(self.config['activation']))
        layers.append(nn.Dropout(self.config.get('dropout', 0.0)))
        
        # Build rest of layers
        for i in range(1, len(self.config['hidden_layers'])):
            prev_size = self.config['hidden_layers'][i-1]
            size = self.config['hidden_layers'][i]
            
            if self.config.get('batch_norm', False):
                layers.append(nn.BatchNorm1d(prev_size))
            
            linear_layer = self.layer_init(
                nn.Linear(prev_size, size),
                std=2 ** 0.5  # Using scalar instead of torch.sqrt(2)
            )
            layers.extend([
                linear_layer,
                self._get_activation(self.config['activation']),
                nn.Dropout(self.config.get('dropout', 0.0))
            ])
            
        return nn.Sequential(*layers)
    
    def _get_activation(self, name):
        return {
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'tanh': nn.Tanh(),
            'linear': nn.Identity()
        }[name.lower()]
    
    def _build_head(self):
        """Build value head"""
        return self.layer_init(
            nn.Linear(self.config['hidden_layers'][-1], 1),
            std=1.0
        )
    
    def forward(self, x, process_state=True):
        features = x
        if process_state:
            # x is a dictionary of features, so concatenate on observation keys
            features = torch.cat([x[key].flatten(start_dim=1) for key in self.observation_keys if key != 'current_period'], dim=-1)
            if 'current_period' in self.observation_keys:
                # Use the current_period value for all rows, so expand it to match the batch size
                current_period_value = x['current_period'].unsqueeze(0).expand(features.size(0), -1).to(self.device)
                # print(current_period_value.shape)
                # print(features.shape)
                features = torch.cat([current_period_value, features], dim=-1)
            # print(features.shape)
            # print(features[0: 2])
        features = self.backbone(features)
        return self.head(features) 