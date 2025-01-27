class Config:
    """Class to manage all configuration settings"""
    def __init__(self, setting_config, hyperparams_config):
        # Store raw configs
        self.setting_config = setting_config
        self.hyperparams_config = hyperparams_config
        
        # Extract all parameters from settings
        self.problem_params = setting_config['problem_params']
        self.store_params = setting_config['store_params']
        self.warehouse_params = setting_config.get('warehouse_params', {})
        self.echelon_params = setting_config.get('echelon_params', {})
        self.observation_params = setting_config['observation_params']
        self.sample_data_params = setting_config['sample_data_params']
        self.params_by_dataset = setting_config['params_by_dataset']
        self.seeds = setting_config.get('seeds')
        self.test_seeds = setting_config.get('test_seeds')
        
        # Extract all hyperparameters
        self.trainer_params = hyperparams_config['trainer_params']
        self.logging_params = hyperparams_config['logging_params']
        self.optimizer_params = hyperparams_config['optimizer_params']
        self.nn_params = hyperparams_config['nn_params']
    
    def get_complete_config(self):
        """Get complete configuration dictionary for logging"""
        return {
            **self.trainer_params,
            'problem_params': self.problem_params,
            'store_params': self.store_params,
            'warehouse_params': self.warehouse_params,
            'echelon_params': self.echelon_params,
            'observation_params': self.observation_params,
            'sample_data_params': self.sample_data_params,
            'params_by_dataset': self.params_by_dataset,
            'seeds': self.seeds,
            'test_seeds': self.test_seeds,
            'optimizer_params': self.optimizer_params,
            'nn_params': self.nn_params
        } 