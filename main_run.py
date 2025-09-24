# main_run.py

# Standard imports
import sys
from collections import defaultdict as DefaultDict
from datetime import datetime
import os
import random
# Core imports
from src import np, torch, yaml
from torch.utils.data import DataLoader

# Environment imports
from src.envs.inventory.env import InventoryEnv
from src.envs.inventory.simulator import Simulator
from src.envs.inventory.hybrid_simulator import HybridSimulator
from src.envs.inventory.range_manager import RangeManager

# Algorithm imports
from src.algorithms.hybrid.losses.loss import PolicyLoss
from src.algorithms.hybrid.agents.hybrid_agent import HybridAgent  # Ensure HybridAgent is imported correctly
from src.algorithms.hybrid.agents.hybrid_agent import (
    GaussianPPOAgent, 
    GumbelSoftmaxAgent, 
    FactoredGumbelSoftmaxAgent,
    ContinuousOnlyAgent,
    FactoredGaussianPPOAgent,
    FactoredHybridAgent
)

# Data handling imports
from src.data.data_handling import DatasetCreator, Dataset, Scenario

# Training imports
from src.training.trainer import Trainer

# Feature registry imports
from src.features.feature_registry import FeatureRegistry
from src.algorithms.hybrid.optimizer_wrappers.hybrid_wrapper import HybridWrapper

from src.utils.config import Config

if True:
    torch.autograd.set_detect_anomaly(True)  # This will help detect anomalies in backward passes
    print('Anomaly detection enabled')

def get_timestamp():
    """Get current timestamp for model saving"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_date_folder():
    """Get folder name based on current date"""
    return datetime.now().strftime("%Y%m%d")

def create_parameter_groups_with_lr(model, base_lr, lr_multipliers):
    """Create parameter groups with different learning rate multipliers"""
    param_groups = []
    
    # Define learning rate multipliers for different components
    # Default to 1.0 if not specified
    value_multiplier = lr_multipliers.get('value', 1.0)
    backbone_multiplier = lr_multipliers.get('backbone', 1.0)
    continuous_multiplier = lr_multipliers.get('continuous', 1.0)
    discrete_multiplier = lr_multipliers.get('discrete', 1.0)
    other_multiplier = lr_multipliers.get('other', 1.0)
    
    # Group parameters by component
    value_params = []
    backbone_params = []
    continuous_params = []
    discrete_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'value' in name:
                value_params.append(param)
            elif 'backbone' in name:
                backbone_params.append(param)
            elif 'continuous' in name:
                continuous_params.append(param)
            elif 'discrete' in name:
                discrete_params.append(param)
            else:
                other_params.append(param)
    
    # Create parameter groups with calculated learning rates
    if value_params:
        param_groups.append({
            'params': value_params, 
            'lr': base_lr * value_multiplier, 
            'name': 'value'
        })
    if backbone_params:
        param_groups.append({
            'params': backbone_params, 
            'lr': base_lr * backbone_multiplier, 
            'name': 'backbone'
        })
    if continuous_params:
        param_groups.append({
            'params': continuous_params, 
            'lr': base_lr * continuous_multiplier, 
            'name': 'continuous'
        })
    if discrete_params:
        param_groups.append({
            'params': discrete_params, 
            'lr': base_lr * discrete_multiplier, 
            'name': 'discrete'
        })
    if other_params:
        param_groups.append({
            'params': other_params, 
            'lr': base_lr * other_multiplier, 
            'name': 'other'
        })
    
    # Print learning rate configuration for verification
    print(f"\n=== LEARNING RATE CONFIGURATION ===")
    print(f"Base learning rate: {base_lr}")
    print(f"Learning rate multipliers: {lr_multipliers}")
    print(f"Final learning rates per group:")
    for group in param_groups:
        print(f"  {group['name']:12}: {group['lr']:.6f} (base_lr × {group['lr']/base_lr:.3f})")
    print("=" * 40)
    
    return param_groups


# ---- DEBUG HOOK: forward NaN/Inf tripwire ----
def register_forward_nan_checks(model, *, name_prefix=""):
    """
    Registers forward hooks on all leaf modules. If any module's output
    or input is non-finite, raises RuntimeError (fail-fast).
    Returns a list of hook handles (remember to .remove() them).
    """
    import torch

    handles = []

    def _check_tensor(t, label, mod_name, mod_cls):
        if isinstance(t, torch.Tensor):
            if not torch.isfinite(t).all():
                bad = (~torch.isfinite(t)).nonzero(as_tuple=False)
                # show up to 5 offending indices for context
                bad_list = bad[:5].tolist() if bad.numel() else []
                msg = (f"[NONFINITE] {label} in {mod_cls} '{mod_name}' "
                       f"shape={tuple(t.shape)} example_idx={bad_list}")
                print(msg)
                raise RuntimeError(msg)

    def _hook(mod, inputs, output):
        mod_name = getattr(mod, "_layer_name", mod.__class__.__name__)
        mod_cls  = mod.__class__.__name__
        # check outputs
        if isinstance(output, (tuple, list)):
            for i, o in enumerate(output):
                _check_tensor(o, f"output[{i}]", mod_name, mod_cls)
        else:
            _check_tensor(output, "output", mod_name, mod_cls)
        # also check inputs (helps catch zero-denoms *before* the next op)
        for i, x in enumerate(inputs):
            _check_tensor(x, f"input[{i}]", mod_name, mod_cls)

    # attach to leaf modules only (modules without children)
    for name, m in model.named_modules():
        if len(list(m.children())) == 0:
            m._layer_name = f"{name_prefix}{name}"
            handles.append(m.register_forward_hook(_hook))
    return handles
# ---- /DEBUG HOOK ----


def run_training(setting_config, hyperparams_config, mode='both'):
    """
    Run training and/or testing with given configurations
    Args:
        setting_config: Configuration for problem settings
        hyperparams_config: Configuration for hyperparameters
        mode: str, one of ['train', 'test', 'both']
    Returns: (train_metrics, test_metrics) - any can be None depending on mode
    """
    # Use device from config, respecting CUDA_VISIBLE_DEVICES
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    
    print(f"Training on device: {device}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    # Update config with correct device
    hyperparams_config['device'] = device

    # Create config object
    config = Config(setting_config, hyperparams_config)

    # Extract parameters from configs
    problem_params = config.setting_config['problem_params']
    store_params = config.setting_config['store_params']
    warehouse_params = config.setting_config['warehouse_params']
    echelon_params = config.setting_config['echelon_params']
    observation_params = config.setting_config['observation_params']
    sample_data_params = config.setting_config['sample_data_params']
    params_by_dataset = config.setting_config['params_by_dataset']
    seeds = config.setting_config['seeds']
    test_seeds = config.setting_config['test_seeds']

    # Extract hyperparameters
    trainer_params = config.hyperparams_config['trainer_params']
    optimizer_params = config.hyperparams_config['optimizer_params']
    nn_params = config.hyperparams_config['nn_params']
    agent_params = config.hyperparams_config['agent_params']

    feature_registry = None
    # Initialize range manager if this is a hybrid problem
    if problem_params.get('is_hybrid', False):
        range_manager = RangeManager(problem_params, device=device)
        network_dims = range_manager.get_network_dimensions()
        print(f'Network dimensions: {network_dims}')
        
        config.hyperparams_config['nn_params']['policy_network']['heads']['discrete'].update({
            'size': network_dims['n_discrete']
        })
        config.hyperparams_config['nn_params']['policy_network']['heads']['continuous'].update({
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
        'agent_params': agent_params
    }

    # Define agent mapping
    agent_mapping = {
        'hybrid': HybridAgent,
        'gaussian_ppo': GaussianPPOAgent,
        'gumbel_softmax': GumbelSoftmaxAgent,
        'factored_gumbel_softmax': FactoredGumbelSoftmaxAgent,
        'continuous_only': ContinuousOnlyAgent,
        'factored_gaussian_ppo': FactoredGaussianPPOAgent,
        'factored_hybrid': FactoredHybridAgent
    }

    # Get agent type from config, default to 'hybrid' if not specified
    agent_type = hyperparams_config.get('agent_params', {}).get('agent_type', 'hybrid')
    
    # Print agent type being used
    print(f"Creating model with agent type: {agent_type}")
    print(f"Creating model with policy: {nn_params['policy_network']['name']}")
    
    if agent_type in agent_mapping:
        model = agent_mapping[agent_type](agent_config, feature_registry, device=device)
        print(f"Created model with agent class: {type(model).__name__}")

        
    else:
        print(f"Warning: Unknown agent type '{agent_type}', defaulting to HybridAgent")
        model = HybridAgent(agent_config, feature_registry, device=device)

    # === DEBUG: forward NaN/Inf tripwires ===
    fwd_nan_handles = []
    try:
        # Whole model (fastest way to get coverage)
        fwd_nan_handles += register_forward_nan_checks(model, name_prefix="model.")
        # If your model exposes submodules, you can be more granular too:
        if hasattr(model, "policy"):
            fwd_nan_handles += register_forward_nan_checks(model.policy, name_prefix="policy.")
        # if hasattr(model, "value_net"):
        #     fwd_nan_handles += register_forward_nan_checks(model.value_net, name_prefix="value.")
        # if hasattr(model, "range_manager"):
        #     fwd_nan_handles += register_forward_nan_checks(model.range_manager, name_prefix="range.")
    except Exception as _e:
        print(f"Failed to register forward NaN checks: {_e}")

    # Get learning rate multipliers from config, with sensible defaults
    lr_multipliers = optimizer_params.get('lr_multipliers', {
        'value': 1.0,
        'backbone': 1.0,
        'continuous': 1.0,
        'discrete': 1.0,
        'other': 1.0
    })

    print(f"Learning rate multipliers from config: {lr_multipliers}")
    print(f"Base learning rate: {optimizer_params['learning_rate']}")

    param_groups = create_parameter_groups_with_lr(
        model, 
        optimizer_params['learning_rate'], 
        lr_multipliers
    )
    optimizer = torch.optim.Adam(param_groups, eps=1e-5)
    
    # optimizer = torch.optim.AdamW(param_groups)

    # optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_params['learning_rate'], eps=1e-5)

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

    # Create optimizer wrapper with PPO params
    wrapper_params = {
        'ppo_params': optimizer_params.get('ppo_params', {})
    }

    optimizer_wrapper = HybridWrapper(model, optimizer, device=device, **wrapper_params)

    # Load previous model if specified
    if trainer_params['load_previous_model']:
        print(f'Loading model from {trainer_params["load_model_path"]}')
        checkpoint = torch.load(trainer_params['load_model_path'], weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Model loaded successfully")

    # Initialize simulator based on type
    simulator = (HybridSimulator(feature_registry, model=model, device=device) 
                if feature_registry else 
                Simulator(device=device))

    # raise an error if the setting_config['problem_params']['setting_name'] is not equal to the hyperparams_config['logging_params']['setting_name']
    if setting_config['problem_params']['setting_name'] != hyperparams_config['logging_params']['setting_name']:
        raise ValueError(f'Setting name in setting_config ({setting_config["problem_params"]["setting_name"]}) does not match setting name in hyperparams_config ({hyperparams_config["logging_params"]["setting_name"]})')

    # We will create a folder for each day of the year, and a subfolder for each model
    trainer_params['save_model_folders'] = [trainer.get_year_month_day(), setting_config['problem_params']['setting_name'], nn_params['policy_network']['name']]

    # We will simply name the model with the current time stamp + a random number between 0 and 1000 to avoid overwriting
    trainer_params['save_model_filename'] = str(trainer.get_time_stamp()) + f'_{random.randint(0, 1000)}'

    # Create loss function
    loss_function = PolicyLoss()

    train_metrics = None
    dev_metrics = None
    test_metrics = None
    
    # Run training if requested
    if mode in ['train', 'both']:
        print('Starting training...')
        train_metrics, dev_metrics = trainer.train(
            trainer_params['epochs'], 
            loss_function, 
            simulator, 
            model, 
            data_loaders, 
            optimizer_wrapper,
            problem_params, 
            observation_params, 
            params_by_dataset, 
            trainer_params,
            config
        )

    # Run test if requested
    if mode in ['test', 'both']:
        print('Starting testing...')
        test_metrics, _ = trainer.test(
            loss_function,
            simulator,
            model,
            data_loaders,
            optimizer_wrapper,
            problem_params,
            observation_params,
            params_by_dataset,
            trainer_params,
            discrete_allocation=store_params['demand']['distribution'] == 'poisson' and False
        )
        print(f'Test total loss: {test_metrics["loss/total"]}')
        print(f'Test reported loss: {test_metrics["loss/reported"]}')

    # Always close logger if it exists
    if trainer.logger is not None:
        trainer.logger.close()
    
    
    # Clean up forward hooks
    for _h in locals().get("fwd_nan_handles", []):
        try:
            _h.remove()
        except Exception:
            pass

    
    return train_metrics, dev_metrics, test_metrics

if __name__ == "__main__":
    # Check if command-line arguments are provided
    if len(sys.argv) == 4:
        setting_name = sys.argv[2]
        hyperparams_name = sys.argv[3]
    elif len(sys.argv) == 2:
        setting_name = 'one_store_lost'
        hyperparams_name = 'vanilla_one_store'
    else:
        print(f'Number of parameters provided including script name: {len(sys.argv)}')
        print(f'Number of parameters should be either 4 or 2')
        assert False

    mode = sys.argv[1]
    if mode not in ['train', 'test', 'both']:
        print(f'Invalid mode: {mode}. Must be one of: train, test, both')
        assert False

    print(f'Setting file name: {setting_name}')
    print(f'Hyperparams file name: {hyperparams_name}\n')

    # Load configs
    config_setting_file = f'configs/settings/{setting_name}.yml'
    config_hyperparams_file = f'configs/policies/{hyperparams_name}.yml'

    with open(config_setting_file, 'r') as file:
        setting_config = yaml.safe_load(file)

    with open(config_hyperparams_file, 'r') as file:
        hyperparams_config = yaml.safe_load(file)

    # Run training/testing with specified mode
    train_metrics, dev_metrics, test_metrics = run_training(setting_config, hyperparams_config, mode=mode)
    
    # Print test metrics if they exist
    if test_metrics is not None:
        print(f'Average per-period test loss: {test_metrics["loss/reported"]}')

