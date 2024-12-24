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
    
    def _build_backbone(self):
        layers = []
        
        # Use LazyLinear for first layer - will infer input size on first forward pass
        layers.append(nn.LazyLinear(self.config['hidden_layers'][0]))
        layers.append(self._get_activation(self.config['activation']))
        layers.append(nn.Dropout(self.config.get('dropout', 0.0)))
        
        # Build rest of layers
        for i in range(1, len(self.config['hidden_layers'])):
            prev_size = self.config['hidden_layers'][i-1]
            size = self.config['hidden_layers'][i]
            
            if self.config.get('batch_norm', False):
                layers.append(nn.BatchNorm1d(prev_size))
            
            layers.extend([
                nn.Linear(prev_size, size),
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
        return nn.Linear(self.config['hidden_layers'][-1], 1)
    
    def forward(self, x, process_state=True):
        features = x
        if process_state:
            # x is a dictionary of features, so concatenate on observation keys
            features = torch.cat([x[key].flatten(start_dim=1) for key in self.observation_keys], dim=-1)
        features = self.backbone(features)
        return self.head(features) 