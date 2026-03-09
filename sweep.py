import wandb
import yaml
import os
import torch
import logging
import time
import tempfile
import json
import fcntl
from typing import List, Optional
from main_run import run_training
from src.utils.path_utils import get_date_folder

# Only import ray if needed (for agent runs)
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    # Create a dummy ray module for the decorator
    class DummyRay:
        @staticmethod
        def remote(*args, **kwargs):
            def decorator(cls):
                return cls
            return decorator
    ray = DummyRay()

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

def _load_top_k_metadata(metadata_path):
    if not os.path.exists(metadata_path):
        return {'entries': []}
    with open(metadata_path, 'r') as f:
        return json.load(f)

def _save_top_k_metadata(metadata_path, data):
    with open(metadata_path, 'w') as f:
        json.dump(data, f, indent=2)

def _atomic_top_k_update(lock_path, update_fn):
    """Lock using flock to safely update top-k metadata across processes."""
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, 'w') as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        update_fn()
        fcntl.flock(lock_file, fcntl.LOCK_UN)

def _format_split_value(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return "[" + ",".join(_format_split_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{...}"
    return str(value)

def _build_split_label(split_by, run_config, setting_config=None):
    if not split_by:
        return None
    if isinstance(split_by, str):
        split_by = [split_by]

    parts = []
    for key in split_by:
        value = run_config.get(key, None)
        if value is None and setting_config is not None:
            value = (
                setting_config
                .get('problem_params', {})
                .get(key)
            )
        if value is None:
            value = "unknown"
        value_str = _format_split_value(value)
        label = f"{key}={value_str}"
        label = label.replace(os.sep, "-").replace(" ", "_")
        parts.append(label)

    return "__".join(parts) if parts else None

def update_sweep_top_k(
    base_dir,
    sweep_id,
    setting_name,
    policy_name,
    run_id,
    best_metric_name,
    best_metric_value,
    best_dev_loss,
    best_dev_loss_reported,
    best_train_loss,
    best_train_loss_reported,
    best_epoch,
    model_state_dict,
    optimizer_state_dict,
    top_k,
    extra_metadata=None,
    split_label=None
):
    if top_k is None or top_k <= 0:
        return
    if model_state_dict is None:
        logging.warning("No model state to save for top-k sweep tracking.")
        return

    sweep_label = sweep_id or "no_sweep"
    top_k_dir = os.path.join(
        base_dir,
        get_date_folder(),
        "sweep_top_k",
        sweep_label,
        setting_name,
        policy_name
    )
    if split_label:
        top_k_dir = os.path.join(top_k_dir, split_label)
    os.makedirs(top_k_dir, exist_ok=True)

    metadata_path = os.path.join(top_k_dir, "top_k_runs.json")
    lock_path = os.path.join(top_k_dir, ".top_k.lock")

    def _update():
        metadata = _load_top_k_metadata(metadata_path)
        entries = metadata.get('entries', [])

        # Remove any existing entry for this run (we'll re-evaluate)
        entries = [e for e in entries if e.get('run_id') != run_id]

        worst_metric = max([e['best_metric_value'] for e in entries], default=None)
        is_candidate = (len(entries) < top_k) or (worst_metric is not None and best_metric_value < worst_metric)

        if not is_candidate:
            metadata['entries'] = sorted(entries, key=lambda x: x['best_dev_loss'])
            _save_top_k_metadata(metadata_path, metadata)
            return

        filename = f"run_{run_id}_{best_metric_name}_{best_metric_value:.6f}.pt"
        checkpoint_path = os.path.join(top_k_dir, filename)

        checkpoint = {
            'epoch': best_epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'best_metric_name': best_metric_name,
            'best_metric_value': best_metric_value,
            'best_dev_loss': best_dev_loss,
            'best_dev_loss_reported': best_dev_loss_reported,
            'best_train_loss': best_train_loss,
            'best_train_loss_reported': best_train_loss_reported,
            'run_id': run_id,
            'sweep_id': sweep_id
        }
        if extra_metadata:
            checkpoint['metadata'] = extra_metadata

        torch.save(checkpoint, checkpoint_path)

        entries.append({
            'run_id': run_id,
            'best_metric_name': best_metric_name,
            'best_metric_value': best_metric_value,
            'best_dev_loss': best_dev_loss,
            'best_dev_loss_reported': best_dev_loss_reported,
            'best_train_loss': best_train_loss,
            'best_train_loss_reported': best_train_loss_reported,
            'best_epoch': best_epoch,
            'checkpoint_path': checkpoint_path
        })

        entries = sorted(entries, key=lambda x: x['best_metric_value'])
        while len(entries) > top_k:
            removed = entries.pop()
            old_path = removed.get('checkpoint_path')
            if old_path and os.path.exists(old_path):
                os.remove(old_path)

        metadata['entries'] = entries
        _save_top_k_metadata(metadata_path, metadata)

    _atomic_top_k_update(lock_path, _update)

# Decorator for ray (only used when running agents)
if RAY_AVAILABLE:
    TrainingWorkerDecorator = ray.remote(num_gpus=1)
else:
    # Dummy decorator if ray is not available
    def TrainingWorkerDecorator(cls):
        return cls

@TrainingWorkerDecorator
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
                'entropy_coef': ('hyperparams', ['optimizer_params', 'ppo_params', 'entropy_coef']),
                'anneal_entropy_coef': ('hyperparams', ['optimizer_params', 'ppo_params', 'anneal_entropy_coef']),
                'min_entropy_coef': ('hyperparams', ['optimizer_params', 'ppo_params', 'min_entropy_coef']),
                'loss_schedule_pathwise': ('hyperparams', ['optimizer_params', 'ppo_params', 'loss_schedule', 'pathwise']),
                'disable_cross_term': ('hyperparams', ['optimizer_params', 'ppo_params', 'disable_cross_term']),
                # Unified temperature parameters
                'initial_temperature': ('hyperparams', ['agent_params', 'initial_temperature']),
                'min_temperature': ('hyperparams', ['agent_params', 'min_temperature']),
                'temperature_decay': ('hyperparams', ['agent_params', 'temperature_decay']),
                'use_straight_through': ('hyperparams', ['agent_params', 'use_straight_through']),
                'add_gumbel_noise': ('hyperparams', ['agent_params', 'add_gumbel_noise']),
                # Add mapping for continuous scale parameter - updated path
                'continuous_scale': ('hyperparams', ['nn_params', 'policy_network', 'continuous_scale']),
                'continuous_shift': ('hyperparams', ['nn_params', 'policy_network', 'continuous_shift']),
                'normalize_by_mean_demand': ('hyperparams', ['nn_params', 'policy_network', 'normalize_by_mean_demand']),
                'single_anchor_normalization': ('hyperparams', ['nn_params', 'policy_network', 'single_anchor_normalization']),
                'discrete_lr_multiplier': ('hyperparams', ['optimizer_params', 'lr_multipliers', 'discrete']),
                'continuous_lr_multiplier': ('hyperparams', ['optimizer_params', 'lr_multipliers', 'continuous']),
                'backbone_lr_multiplier': ('hyperparams', ['optimizer_params', 'lr_multipliers', 'backbone']),
                'value_lr_multiplier': ('hyperparams', ['optimizer_params', 'lr_multipliers', 'value']),
                'other_lr_multiplier': ('hyperparams', ['optimizer_params', 'lr_multipliers', 'other']),
                'hidden_layers': ('hyperparams', ['nn_params', 'policy_network', 'hidden_layers']),
                'use_wandb': ('hyperparams', ['logging_params', 'use_wandb']),
                'single_anchor_reward_scaling': ('hyperparams', ['optimizer_params', 'ppo_params', 'single_anchor_reward_scaling']),
                # Add mapping for threshold parameter
                'fixed_ordering_cost_threshold': ('setting', ['problem_params', 'discrete_features', 'fixed_ordering_cost', 'thresholds', 1]),
                'fixed_cost': ('setting', ['problem_params', 'discrete_features', 'fixed_ordering_cost', 'values', 1]),
                'n_stores': ('setting', ['problem_params', 'n_stores']),
                'sweep_top_k_split_by': ('hyperparams', ['trainer_params', 'sweep_top_k_split_by']),
                'sweep_top_k_metric': ('hyperparams', ['trainer_params', 'sweep_top_k_metric']),
            }
            
            # Update configs based on sweep parameters from run.config
            for param_name, param_value in run.config.items():
                if param_name in param_mappings:
                    config_type, param_path = param_mappings[param_name]
                    target_config = hyperparams_config if config_type == 'hyperparams' else setting_config
                    
                    # Navigate to the correct nested dict/list
                    current = target_config
                    for i, key in enumerate(param_path[:-1]):
                        if isinstance(current, dict):
                            if key not in current:
                                next_key = param_path[i + 1]
                                current[key] = {} if isinstance(next_key, str) else []
                            current = current[key]
                        elif isinstance(current, list):
                            # Ensure list is long enough
                            while len(current) <= key:
                                current.append(None)
                            if i < len(param_path) - 2:
                                if current[key] is None:
                                    current[key] = {} if isinstance(param_path[i + 1], str) else []
                            current = current[key]
                    
                    # Set the final value
                    if isinstance(current, dict):
                        current[param_path[-1]] = param_value
                    elif isinstance(current, list):
                        # Ensure list is long enough
                        while len(current) <= param_path[-1]:
                            current.append(None)
                        current[param_path[-1]] = param_value

                    # Mirror shared hidden layers to value network for sweeps
                    if param_name == 'hidden_layers' and config_type == 'hyperparams':
                        value_network = (
                            hyperparams_config
                            .get('nn_params', {})
                            .get('value_network', {})
                        )
                        if isinstance(value_network, dict):
                            value_network['hidden_layers'] = param_value

            # Scale fixed ordering cost by number of stores after overrides
            problem_params = setting_config.get('problem_params', {})
            n_stores = problem_params.get('n_stores', 1)
            fixed_cost_config = (
                problem_params
                .get('discrete_features', {})
                .get('fixed_ordering_cost', {})
            )
            if fixed_cost_config.get('_scaled_by_n_stores'):
                print("Fixed ordering cost already scaled by n_stores; skipping.")
            else:
                values = fixed_cost_config.get('values')
                if isinstance(values, list):
                    fixed_cost_config['values'] = [
                        (v * n_stores) if isinstance(v, (int, float)) else v
                        for v in values
                    ]
                    fixed_cost_config['_scaled_by_n_stores'] = True
                    print(f"Scaled fixed_ordering_cost values by n_stores={n_stores}: {fixed_cost_config['values']}")

            # Keep wandb config in sync with updated setting config
            try:
                wandb.config.update({'setting_config': setting_config}, allow_val_change=True)
            except Exception as e:
                print(f"Warning: failed to update wandb setting_config: {e}")

            # Add relevant problem parameters as tags
            problem_params = setting_config.get('problem_params', {})
            wandb.config.tags = [
                f"stores_{problem_params.get('n_stores', 0)}",
                f"warehouses_{problem_params.get('n_warehouses', 0)}",
                f"echelons_{problem_params.get('n_extra_echelons', 0)}",
                'hybrid' if problem_params.get('is_hybrid', False) else 'single',
                'profit' if problem_params.get('maximize_profit', False) else 'cost',
                'lost_demand' if problem_params.get('lost_demand', False) else 'backorder',
                hyperparams_config['agent_params']['agent_type']
            ]

            # Run training
            train_metrics, dev_metrics, test_metrics, best_performance = run_training(
                setting_config,
                hyperparams_config,
                mode='both',
                return_best_state=True
            )

            # Save top-k sweep runs by best dev loss (if configured)
            trainer_params = hyperparams_config.get('trainer_params', {})
            top_k = trainer_params.get('sweep_top_k', 0)
            base_dir = trainer_params.get('base_dir', 'models/saved_models')
            setting_name = setting_config.get('problem_params', {}).get('setting_name', 'unknown_setting')
            policy_name = hyperparams_config.get('nn_params', {}).get('policy_network', {}).get('name', 'unknown_policy')
            split_label = _build_split_label(
                trainer_params.get('sweep_top_k_split_by'),
                run.config,
                setting_config=setting_config
            )
            metric_name = trainer_params.get('sweep_top_k_metric', 'dev_loss_reported')
            metric_value = best_performance.get(metric_name)
            if metric_value is None:
                metric_value = best_performance.get('dev_loss_reported')
            if metric_value is None:
                metric_value = best_performance.get('dev_loss', float('inf'))

            update_sweep_top_k(
                base_dir=base_dir,
                sweep_id=run.sweep_id,
                setting_name=setting_name,
                policy_name=policy_name,
                run_id=run.id,
                best_metric_name=metric_name,
                best_metric_value=metric_value,
                best_dev_loss=best_performance.get('dev_loss', float('inf')),
                best_dev_loss_reported=best_performance.get('dev_loss_reported', float('inf')),
                best_train_loss=best_performance.get('train_loss', float('inf')),
                best_train_loss_reported=best_performance.get('train_loss_reported', float('inf')),
                best_epoch=best_performance.get('best_epoch'),
                model_state_dict=best_performance.get('model_params_to_save'),
                optimizer_state_dict=best_performance.get('optimizer_state_to_save'),
                top_k=top_k,
                extra_metadata={
                    'config_files': config_files,
                    'wandb_name': run.name
                },
                split_label=split_label
            )
            
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

def create_sweep_config(agent_sweep_file, setting_sweep_file):
    """Create sweep configuration by merging agent and setting sweep configs."""
    # Load the sweep configs
    with open(agent_sweep_file, 'r') as file:
        agent_sweep_config = yaml.safe_load(file)
    with open(setting_sweep_file, 'r') as file:
        setting_sweep_config = yaml.safe_load(file)
    
    # Get the config files from the sweep configs
    config_files = {
        'setting': setting_sweep_config.get('config_files', {}).get('setting'),
        'hyperparams': agent_sweep_config.get('config_files', {}).get('hyperparams')
    }
    
    # Load the full configs
    with open(config_files['setting'], 'r') as file:
        setting_config = yaml.safe_load(file)
    with open(config_files['hyperparams'], 'r') as file:
        hyperparams_config = yaml.safe_load(file)
    
    # Create base sweep config
    sweep_config = {
        'method': agent_sweep_config.get('method', 'random'),
        'metric': agent_sweep_config.get('metric', {
            'name': 'dev/loss/best',
            'goal': 'minimize'
        }),
        'parameters': {
            'config_files': {
                'value': config_files
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
    
    # Merge parameters from both sweep configs
    for param_name, param_config in agent_sweep_config.get('parameters', {}).items():
        if param_name not in ['config_files', 'setting_config', 'hyperparams_config']:
            sweep_config['parameters'][param_name] = param_config
            
    for param_name, param_config in setting_sweep_config.get('parameters', {}).items():
        if param_name not in ['config_files', 'setting_config', 'hyperparams_config']:
            sweep_config['parameters'][param_name] = param_config
    
    return sweep_config

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--create', action='store_true', help='Create a new sweep')
    parser.add_argument('--agent', action='store_true', help='Run sweep agent(s)')
    parser.add_argument('--count', type=int, default=1, help='Number of runs per GPU')
    parser.add_argument('--gpus', nargs='+', type=int, required=True, help='List of GPU IDs to use')
    parser.add_argument('--agent_sweep', type=str, default='configs/sweeps/agents/hybrid.yml', 
                        help='Agent sweep configuration file')
    parser.add_argument('--setting_sweep', type=str, default='configs/sweeps/settings/fixed_costs.yml', 
                        help='Setting sweep configuration file')
    args = parser.parse_args()

    if args.create:
        # Create sweep config by merging agent and setting sweep configs
        sweep_config = create_sweep_config(args.agent_sweep, args.setting_sweep)
        
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
                    "CUDA_VISIBLE_DEVICES": ",".join(map(str, available_gpus)),
                    # "CUDA_LAUNCH_BLOCKING": "1",
                    # "TORCH_USE_CUDA_DSA": "1"
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