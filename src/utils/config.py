class Config:
    """Class to manage all configuration settings"""
    def __init__(self, setting_config, hyperparams_config):
        # Store raw configs only
        self.setting_config = setting_config
        self.hyperparams_config = hyperparams_config
    
    def get_complete_config(self):
        """Get complete configuration dictionary for logging"""
        return {
            'setting_config': self.setting_config,
            'hyperparams_config': self.hyperparams_config
        } 