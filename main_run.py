# main_run.py

# Standard imports
import sys
from collections import defaultdict as DefaultDict
import logging
from datetime import datetime

# Core imports
from src import np, torch, yaml
from torch.utils.data import DataLoader

# Environment imports
from src.envs.inventory.env import InventoryEnv
from src.envs.inventory.simulator import Simulator
from src.envs.inventory.hybrid_simulator import HybridSimulator
from src.envs.inventory.range_manager import RangeManager

# Algorithm imports
from src.algorithms.common.policies.policy import PolicyNetwork
from src.algorithms.hdpo.losses.loss import PolicyLoss
from src.algorithms.hdpo.agents.hdpo_agent import HDPOAgent  # Ensure HDPOAgent is imported correctly
from src.algorithms.hybrid.agents.hybrid_agent import HybridAgent  # Ensure HybridAgent is imported correctly

# Data handling imports
from src.data.data_handling import DatasetCreator, Dataset, Scenario

# Training imports
from src.training.trainer import Trainer

# Feature registry imports
from src.features.feature_registry import FeatureRegistry
# from src.algorithms.optimizers.hdpo_optimizer import HDPOOptimizer
# from src.algorithms.optimizers.hybrid_optimizer import HybridOptimizer
from src.algorithms.hdpo.optimizer_wrappers.hdpo_wrapper import HDPOWrapper
from src.algorithms.hybrid.optimizer_wrappers.hybrid_wrapper import HybridWrapper

from src.utils.config_loader import ConfigLoader
from src.utils.logging_config import setup_logging

torch.autograd.set_detect_anomaly(True)  # This will help detect anomalies in backward passes

def get_timestamp():
    """Get current timestamp for model saving"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_date_folder():
    """Get folder name based on current date"""
    return datetime.now().strftime("%Y%m%d")

# Check if command-line arguments are provided
if len(sys.argv) == 4:
    setting_name = sys.argv[2]
    hyperparams_name = sys.argv[3]
elif len(sys.argv) == 2:
    setting_name = 'one_store_lost'
    hyperparams_name = 'vanilla_one_store'
else:
    print(f'Number of parameters provided including script name: {len(sys.argv)}')
    print(f'Number of parameters should be either 4 or 2 (so that last 2 parameters defined in main_run.py)')
    assert False

train_or_test = sys.argv[1]

print(f'Setting file name: {setting_name}')
print(f'Hyperparams file name: {hyperparams_name}\n')

# Set device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load configs
config_setting_file = f'configs/settings/{setting_name}.yml'
config_hyperparams_file = f'configs/policies/{hyperparams_name}.yml'

with open(config_setting_file, 'r') as file:
    config_setting = yaml.safe_load(file)

with open(config_hyperparams_file, 'r') as file:
    config_hyperparams = yaml.safe_load(file)

# Extract parameters from configs
problem_params = config_setting['problem_params']
store_params = config_setting['store_params']
warehouse_params = config_setting.get('warehouse_params', {})
echelon_params = config_setting.get('echelon_params', {})
observation_params = config_setting['observation_params']
sample_data_params = config_setting['sample_data_params']
params_by_dataset = config_setting['params_by_dataset']
seeds = config_setting.get('seeds')
test_seeds = config_setting.get('test_seeds')

# Extract hyperparameters
trainer_params = config_hyperparams['trainer_params']
optimizer_params = config_hyperparams['optimizer_params']
nn_params = config_hyperparams['nn_params']

feature_registry = None
# Initialize range manager if this is a hybrid problem
if problem_params.get('is_hybrid', False):
    range_manager = RangeManager(problem_params, device=device)
    network_dims = range_manager.get_network_dimensions()
    print(f'Network dimensions: {network_dims}')
    
    config_hyperparams['nn_params']['policy_network']['heads']['discrete'].update({
        'size': network_dims['n_discrete']
    })
    config_hyperparams['nn_params']['policy_network']['heads']['continuous'].update({
        'size': network_dims['n_continuous']
    })
    feature_registry = FeatureRegistry(nn_params, range_manager)

# Create datasets based on split type
if sample_data_params['split_by_period']:
    scenario = Scenario(
        periods=None,
        problem_params=problem_params,
        store_params=store_params,
        warehouse_params=warehouse_params,
        echelon_params=echelon_params,
        num_samples=params_by_dataset['train']['n_samples'],
        observation_params=observation_params,
        seeds=seeds
    )
    
    dataset_creator = DatasetCreator()
    train_dataset, dev_dataset, test_dataset = dataset_creator.create_datasets(
        scenario,
        split=True,
        by_period=True,
        periods_for_split=[sample_data_params[k] for k in ['train_periods', 'dev_periods', 'test_periods']]
    )
else:
    # For synthetic data
    scenario = Scenario(
        periods=params_by_dataset['train']['periods'],
        problem_params=problem_params,
        store_params=store_params,
        warehouse_params=warehouse_params,
        echelon_params=echelon_params,
        num_samples=params_by_dataset['train']['n_samples'] + params_by_dataset['dev']['n_samples'],
        observation_params=observation_params,
        seeds=seeds
    )
    
    dataset_creator = DatasetCreator()
    train_dataset, dev_dataset = dataset_creator.create_datasets(
        scenario,
        split=True,
        by_sample_indexes=True,
        sample_index_for_split=params_by_dataset['dev']['n_samples']
    )
    
    # Create separate test scenario
    test_scenario = Scenario(
        periods=params_by_dataset['test']['periods'],
        problem_params=problem_params,
        store_params=store_params,
        warehouse_params=warehouse_params,
        echelon_params=echelon_params,
        num_samples=params_by_dataset['test']['n_samples'],
        observation_params=observation_params,
        seeds=test_seeds
    )
    
    test_dataset = dataset_creator.create_datasets(test_scenario, split=False)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=params_by_dataset['train']['batch_size'], shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=params_by_dataset['dev']['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=params_by_dataset['test']['batch_size'], shuffle=False)
data_loaders = {'train': train_loader, 'dev': dev_loader, 'test': test_loader}

# Create agent config
agent_config = {
    'scenario': scenario,
    'nn_params': nn_params,
}

# Create model and optimizer
if feature_registry:
    model = HybridAgent(agent_config, feature_registry, device=device)
else:
    model = HDPOAgent(agent_config, device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_params['learning_rate'])

# Initialize trainer
trainer = Trainer(device=device)

# Use trainer params directly from config, with fallbacks
if not trainer_params.get('save_model_folders'):
    trainer_params['save_model_folders'] = [
        get_date_folder(),
        nn_params['policy_network']['name']
    ]

if not trainer_params.get('save_model_filename'):
    trainer_params['save_model_filename'] = get_timestamp()

# Create optimizer wrapper
wrapper_params = {
    'gradient_clip': config_hyperparams['optimizer_params'].get('gradient_clip'),
    'weight_decay': config_hyperparams['optimizer_params'].get('weight_decay', 0.0)
}

optimizer_wrapper = (
    HybridWrapper(model, optimizer, device=device, **wrapper_params)
    if feature_registry else
    HDPOWrapper(model, optimizer, device=device)
)

# Load previous model if specified
if trainer_params['load_previous_model']:
    print(f'Loading model from {trainer_params["load_model_path"]}')
    checkpoint = torch.load(trainer_params['load_model_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Initialize simulator based on type
simulator = (HybridSimulator(feature_registry, device=device) 
            if feature_registry else 
            Simulator(device=device))

# We will create a folder for each day of the year, and a subfolder for each model
trainer_params['save_model_folders'] = [trainer.get_year_month_day(), nn_params['policy_network']['name']]

# We will simply name the model with the current time stamp
trainer_params['save_model_filename'] = trainer.get_time_stamp()

#create loss function
loss_function = PolicyLoss()

if train_or_test == 'train':
    trainer.train(
        trainer_params['epochs'], 
        loss_function, 
        simulator, 
        model, 
        data_loaders, 
        optimizer_wrapper,  # Pass optimizer_wrapper instead of optimizer
        problem_params, 
        observation_params, 
        params_by_dataset, 
        trainer_params
    )

elif train_or_test == 'test':
    # Deploy on test set, and enforce discrete allocation if the demand is poisson
    average_test_loss, average_test_loss_to_report = trainer.test(
        loss_function, 
        simulator, 
        model, 
        data_loaders, 
        optimizer, 
        problem_params, 
        observation_params, 
        params_by_dataset, 
        discrete_allocation=store_params['demand']['distribution'] == 'poisson'
    )

    print(f'Average per-period test loss: {average_test_loss_to_report}')

else:
    print(f'Invalid argument: {train_or_test}')
    assert False

# Set up logging
logger = setup_logging(level=logging.INFO)

# Load and merge configs
config_loader = ConfigLoader()
config_hyperparams = config_loader.load_config(config_hyperparams_file)

logger.info(f"Loaded configuration: {config_hyperparams}")