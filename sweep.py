import wandb
import yaml
import os
import torch
from main_run import run_training

def save_sweep_id(sweep_id, filename='sweep_id.txt'):
    """Save sweep_id to a file"""
    with open(filename, 'w') as f:
        f.write(sweep_id)

def load_sweep_id(filename='sweep_id.txt'):
    """Load sweep_id from a file"""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return f.read().strip()
    return None

def get_available_gpus():
    """Get list of available GPU indices"""
    return list(range(torch.cuda.device_count()))

def parse_gpu_arg(gpu_arg):
    """
    Parse GPU argument to return list of GPU indices
    Args:
        gpu_arg: Can be:
            - List of integers [0, 2, 3]
            - String "available" or "all" for all available GPUs
            - Single integer for one GPU
    Returns:
        List of GPU indices
    """
    if gpu_arg is None:
        return [None]  # Use CPU
    
    if isinstance(gpu_arg, str):
        if gpu_arg.lower() in ["available", "all"]:
            return get_available_gpus()
        # Handle comma-separated string
        try:
            return [int(idx) for idx in gpu_arg.split(',')]
        except ValueError:
            raise ValueError(f"Invalid GPU specification: {gpu_arg}")
    
    if isinstance(gpu_arg, int):
        return [gpu_arg]
    
    if isinstance(gpu_arg, list):
        return gpu_arg
    
    raise ValueError(f"Invalid GPU specification: {gpu_arg}")

def run_agent_on_gpu(sweep_id, gpu_idx, count):
    """Run a sweep agent on a specific GPU"""
    if gpu_idx is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        print(f"Starting agent on GPU {gpu_idx}")
    else:
        print("Starting agent on CPU")
    
    wandb.agent(sweep_id, train_sweep, count=count)

# Define the sweep configuration
sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'dev/loss/reported',  # This matches the metric name from your logger
        'goal': 'minimize'
    },
    'parameters': {
        'config_files': {
            'value': {
                'setting': 'configs/settings/hybrid_general.yml',
                'hyperparams': 'configs/policies/hybrid_general_policy.yml'
            }
        },
        'hyperparams.optimizer_params.learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,  # 0.00001
            'max': 1e-2   # 0.01
        },
        'hyperparams.optimizer_params.ppo_params.num_epochs': {
            'values': [5, 10, 15]
        },
        'hyperparams.optimizer_params.ppo_params.value_function_coef': {
            'distribution': 'uniform',
            'min': 0.3,
            'max': 0.7
        },
        'hyperparams.optimizer_params.ppo_params.gamma': {
            'distribution': 'uniform',
            'min': 0.9,
            'max': 0.99
        },
        'hyperparams.optimizer_params.ppo_params.gae_lambda': {
            'distribution': 'uniform',
            'min': 0.9,
            'max': 0.99
        }
    }
}

def train_sweep():
    with wandb.init() as run:
        # Get the files from the config
        config_files = wandb.config.config_files
        
        # Load the base configs
        with open(config_files['setting'], 'r') as f:
            setting_config = yaml.safe_load(f)
        with open(config_files['hyperparams'], 'r') as f:
            hyperparams_config = yaml.safe_load(f)
            
        # Update the hyperparams config with sweep values
        for key, value in wandb.config.items():
            if key != 'config_files':  # Skip the config files entry
                # Split the key into config type and path
                config_type, *path = key.split('.')
                
                # Update the appropriate config
                current = setting_config if config_type == 'setting' else hyperparams_config
                
                # Navigate to the correct nested location
                for p in path[:-1]:
                    current = current[p]
                
                # Update the value
                current[path[-1]] = value
        
        # Run training and testing
        train_metrics, test_metrics = run_training(setting_config, hyperparams_config)
        
        # Log test metrics
        wandb.log({
            "test/loss/total": test_metrics["loss/total"],
            "test/loss/reported": test_metrics["loss/reported"]
        })

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--create', action='store_true', help='Create a new sweep')
    parser.add_argument('--agent', action='store_true', help='Run sweep agent(s)')
    parser.add_argument('--count', type=int, default=1, help='Number of runs per agent')
    parser.add_argument('--gpu', type=str, help='GPU specification. Can be "all", "available", a single number, or a comma-separated list (e.g., "0,2,3")')
    args = parser.parse_args()

    if args.create:
        # Create a new sweep and save the ID
        sweep_id = wandb.sweep(sweep_config, project="inventory_control")
        save_sweep_id(sweep_id)
        print(f"Created sweep with ID: {sweep_id}")
    
    if args.agent:
        # Load existing sweep ID
        sweep_id = load_sweep_id()
        if sweep_id is None:
            raise ValueError("No sweep ID found. Create a sweep first using --create")
        
        # Parse GPU argument
        gpu_indices = parse_gpu_arg(args.gpu)
        
        # Import multiprocessing here to avoid potential issues with CUDA initialization
        import multiprocessing as mp
        
        # Create a process for each GPU
        processes = []
        for gpu_idx in gpu_indices:
            p = mp.Process(
                target=run_agent_on_gpu,
                args=(sweep_id, gpu_idx, args.count)
            )
            p.start()
            processes.append(p)
        
        # Wait for all processes to complete
        for p in processes:
            p.join() 