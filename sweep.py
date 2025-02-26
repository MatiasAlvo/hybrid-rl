import wandb
import yaml
import os
import torch
import ray
import logging
import time
import tempfile
from typing import List, Optional
from main_run import run_training

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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

@ray.remote(num_gpus=1)  # Explicitly request 1 GPU for each worker
class TrainingWorker:
    def __init__(self, gpu_id: int):
        """Initialize worker with specific GPU"""
        try:
            # Set the GPU for this worker
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            self.gpu_id = gpu_id
            
            # Force torch to reinitialize CUDA
            torch.cuda.empty_cache()
            
            # Verify CUDA is available
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA not available after setting GPU {gpu_id}")
            
            # Set device and verify
            torch.cuda.set_device(0)  # Use the first (and only) visible GPU
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            logging.info(f"Worker initialized on GPU {gpu_id} ({device_name})")
            
        except Exception as e:
            logging.error(f"Failed to initialize GPU {gpu_id}: {str(e)}")
            raise
    
    def run_sweep(self, sweep_id: str):
        """Run a single sweep trial"""
        try:
            # Verify GPU is still properly set
            if not torch.cuda.is_available():
                raise RuntimeError(f"GPU {self.gpu_id} not available for sweep")
            
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            pid = os.getpid()
            logging.info(f"Process {pid} running sweep on GPU {self.gpu_id} ({device_name})")
            
            wandb.agent(
                sweep_id,
                function=lambda: train_sweep(wandb.config),  # Pass wandb.config explicitly
                count=1,
                project="inventory_control"
            )
            return True
            
        except Exception as e:
            logging.error(f"Error in sweep on GPU {self.gpu_id}: {str(e)}")
            raise

def train_sweep(sweep_config):
    """Run a single sweep trial"""
    try:
        # Initialize wandb first
        run = wandb.init(
            project="inventory_control",
            config=sweep_config
        )
        
        # Get config_files directly from run.config
        config_files = run.config['config_files']
        
        try:
            # Load the full configs from the files
            with open(config_files['setting'], 'r') as file:
                setting_config = yaml.safe_load(file)
            with open(config_files['hyperparams'], 'r') as file:
                hyperparams_config = yaml.safe_load(file)

            # Define parameter mappings with their config destinations
            param_mappings = {
                'learning_rate': ('hyperparams', ['optimizer_params', 'learning_rate']),
                'anneal_lr': ('hyperparams', ['optimizer_params', 'anneal_lr']),
                'num_epochs': ('hyperparams', ['optimizer_params', 'ppo_params', 'num_epochs']),
                'value_function_coef': ('hyperparams', ['optimizer_params', 'ppo_params', 'value_function_coef']),
                'gamma': ('hyperparams', ['optimizer_params', 'ppo_params', 'gamma']),
                'gae_lambda': ('hyperparams', ['optimizer_params', 'ppo_params', 'gae_lambda']),
                'clip_coef': ('hyperparams', ['optimizer_params', 'ppo_params', 'clip_coef']),
                'normalize_advantages': ('hyperparams', ['optimizer_params', 'ppo_params', 'normalize_advantages']),
                'use_gae': ('hyperparams', ['optimizer_params', 'ppo_params', 'use_gae']),
                'policy_activation': ('hyperparams', ['nn_params', 'policy_network', 'activation']),
                'value_activation': ('hyperparams', ['nn_params', 'value_network', 'activation']),
                'normalize_observations': ('setting', ['observation_params', 'normalize_observations']),
                'reward_scaling': ('hyperparams', ['optimizer_params', 'ppo_params', 'reward_scaling']),
                'buffer_periods': ('hyperparams', ['optimizer_params', 'ppo_params', 'buffer_periods']),
                'pathwise_coef': ('hyperparams', ['optimizer_params', 'ppo_params', 'pathwise_coef']),
                'reward_scaling_pathwise': ('hyperparams', ['optimizer_params', 'ppo_params', 'reward_scaling_pathwise']),
                'max_grad_norm': ('hyperparams', ['optimizer_params', 'ppo_params', 'max_grad_norm']),
            }
            
            # Update configs based on sweep parameters from run.config
            for param_name, param_value in run.config.items():
                if param_name in param_mappings:
                    config_type, param_path = param_mappings[param_name]
                    target_config = hyperparams_config if config_type == 'hyperparams' else setting_config
                    
                    # Navigate to the correct nested dict
                    current_dict = target_config
                    for key in param_path[:-1]:
                        if key not in current_dict:
                            current_dict[key] = {}
                        current_dict = current_dict[key]
                    current_dict[param_path[-1]] = param_value

            # Add relevant problem parameters as tags
            problem_params = setting_config.get('problem_params', {})
            wandb.config.tags = [
                f"stores_{problem_params.get('n_stores', 0)}",
                f"warehouses_{problem_params.get('n_warehouses', 0)}",
                f"echelons_{problem_params.get('n_extra_echelons', 0)}",
                'hybrid' if problem_params.get('is_hybrid', False) else 'single',
                'profit' if problem_params.get('maximize_profit', False) else 'cost',
                'lost_demand' if problem_params.get('lost_demand', False) else 'backorder'
            ]

            # Run training
            train_metrics, dev_metrics, test_metrics = run_training(setting_config, hyperparams_config, mode='both')
            
        except Exception as e:
            print(f"Sweep run failed with error: {str(e)}")
            logging.error(f"Sweep run failed with error: {str(e)}", exc_info=True)
            raise
        finally:
            wandb.finish()
            
    except Exception as e:
        print(f"Failed to initialize wandb: {str(e)}")
        logging.error(f"Failed to initialize wandb: {str(e)}", exc_info=True)
        raise

def flatten_dict(d, parent_key='', sep='.'):
    """Flatten a nested dictionary by concatenating keys with a separator."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def create_sweep_config(config_files):
    """Create sweep configuration with all hyperparameters from config files."""
    # Load the full configs
    with open(config_files['setting'], 'r') as file:
        setting_config = yaml.safe_load(file)
    with open(config_files['hyperparams'], 'r') as file:
        hyperparams_config = yaml.safe_load(file)
    
    # Create base sweep config
    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'dev/loss/best',
            'goal': 'minimize'
        },
        'parameters': {
            'config_files': {
                'value': config_files
            },
            # Original parameters
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-4,
                'max': 1e-3
            },
            'anneal_lr': {
                'values': [True]
                # 'values': [True, False]
            },
            'num_epochs': {
                # 'values': [1]
                'values': [5]
            },
            'value_function_coef': {
                'distribution': 'log_uniform_values',
                'min': 0.1,
                'max': 0.2
            },
            'gamma': {
                'distribution': 'uniform',
                'min': 0.92,
                'max': 0.95
            },
            'gae_lambda': {
                'distribution': 'uniform',
                'min': 0.95,
                'max': 0.97
            },
            'clip_coef': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 0.2
            },
            # New parameters
            'policy_activation': {
                'values': ['ELU']
                # 'values': ['ELU']
                # 'values': ['Tanh', 'ReLU', 'ELU']
            },
            'value_activation': {
                'values': ['ELU']
                # 'values': ['ELU']
                # 'values': ['Tanh', 'ReLU', 'ELU']
            },
            'normalize_advantages': {
                'values': [True]
                # 'values': [False, True]
            },
            'use_gae': {
                'values': [True]
                # 'values': [True, False]
            },
            'normalize_observations': {
                # 'values': [False, True]
                'values': [False]
            },
            'reward_scaling': {
                'values': [True]
                # 'values': [False, True]
            },
            'buffer_periods': {
                'values': [50]  # Adjust range as needed
                # 'values': [0, 20, 50]  # Adjust range as needed
            },
            'pathwise_coef': {
                'distribution': 'uniform',
                'min': 8.0,
                'max': 10.0
            },
            'reward_scaling_pathwise': {
                'values': [True]
                # 'values': [True, False]
            },
            'max_grad_norm': {
                'distribution': 'uniform',
                'min': 5.0,
                'max': 10.0
            },
            # Store complete original configs
            'setting_config': {
                'value': setting_config
            },
            'hyperparams_config': {
                'value': hyperparams_config
            }
        }
    }
    
    return sweep_config

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--create', action='store_true', help='Create a new sweep')
    parser.add_argument('--agent', action='store_true', help='Run sweep agent(s)')
    parser.add_argument('--count', type=int, default=1, help='Number of runs per GPU')
    parser.add_argument('--gpus', nargs='+', type=int, required=True, help='List of GPU IDs to use')
    args = parser.parse_args()

    config_files = {
        'setting': 'configs/settings/hybrid_general.yml',
        'hyperparams': 'configs/policies/hybrid_general_policy.yml'
    }

    if args.create:
        # Create sweep config with all parameters
        sweep_config = create_sweep_config(config_files)
        
        sweep_id = wandb.sweep(
            sweep_config, 
            project="inventory_control"
        )
        save_sweep_id(sweep_id)
        print(f"Created sweep with ID: {sweep_id}")
    
    if args.agent:
        # Load the sweep ID
        sweep_id = load_sweep_id()
        if sweep_id is None:
            raise ValueError("No sweep ID found. Please create a sweep first using --create")
            
        # Start timing
        start_time = time.time()
        
        # Verify GPU availability first
        available_gpus = []
        for gpu_id in args.gpus:
            try:
                with torch.cuda.device(gpu_id):
                    torch.cuda.get_device_name(gpu_id)
                available_gpus.append(gpu_id)
            except Exception as e:
                logging.warning(f"GPU {gpu_id} not available: {str(e)}")
        
        if not available_gpus:
            raise RuntimeError("No requested GPUs are available")
        
        logging.info(f"Available GPUs: {available_gpus}")
        
        # Initialize Ray with explicit GPU configuration
        ray.init(
            num_cpus=len(available_gpus),
            num_gpus=len(available_gpus),
            include_dashboard=False,
            ignore_reinit_error=True,
            logging_level=logging.ERROR,
            _temp_dir=tempfile.mkdtemp(),
            runtime_env={
                "env_vars": {
                    "CUDA_VISIBLE_DEVICES": ",".join(map(str, available_gpus))
                }
            }
        )
        
        # Create workers only for available GPUs
        workers = []
        for gpu_id in available_gpus:
            worker = TrainingWorker.remote(gpu_id)
            workers.append(worker)
        
        logging.info(f"Created workers for GPUs: {available_gpus}")
        
        # Run sweeps
        try:
            futures = []
            total_runs = len(available_gpus) * args.count
            completed_runs = 0
            
            # Launch initial batch of runs
            for worker in workers:
                futures.append(worker.run_sweep.remote(sweep_id))
                completed_runs += 1
            
            # Keep launching new runs as they complete
            while completed_runs < total_runs:
                # Wait for any run to complete
                done_id, futures = ray.wait(futures, num_returns=1)
                
                # Launch next run on any available worker
                if completed_runs < total_runs:
                    # Round-robin worker selection
                    worker_idx = completed_runs % len(workers)
                    futures.append(workers[worker_idx].run_sweep.remote(sweep_id))
                    completed_runs += 1
                    logging.info(f"Completed {completed_runs}/{total_runs} runs")
            
            # Wait for remaining runs to complete
            ray.get(futures)
            
        except KeyboardInterrupt:
            logging.info("\nGracefully shutting down...")
        except Exception as e:
            logging.error(f"Error during sweep execution: {e}")
        finally:
            ray.shutdown()
            
        total_time = time.time() - start_time
        logging.info(f"Sweep completed in {total_time:.2f} seconds")