import yaml
from copy import deepcopy

class ConfigLoader:
    def __init__(self):
        self.base_config = self._load_base_config()
    
    def _load_base_config(self):
        """Load base configuration file"""
        with open('configs/policies/base_config.yml', 'r') as f:
            return yaml.safe_load(f)
    
    def load_config(self, config_name):
        """Load specific config and merge with base config"""
        # Start with a deep copy of base config
        final_config = deepcopy(self.base_config)
        
        # Load specific config
        with open(f'configs/policies/{config_name}.yml', 'r') as f:
            specific_config = yaml.safe_load(f)
        
        # Recursively merge configs
        self._merge_configs(final_config, specific_config)
        
        return final_config
    
    def _merge_configs(self, base, specific):
        """Recursively merge specific config into base config"""
        for key, value in specific.items():
            if isinstance(value, dict) and key in base:
                self._merge_configs(base[key], value)
            else:
                base[key] = value 