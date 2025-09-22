import wandb
import os

# Force offline mode
os.environ['WANDB_MODE'] = 'offline'

try:
    # Test basic init (should work offline)
    wandb.init(project="test", mode="offline")
    print("✓ wandb.init() works offline")
    
    # Test sweep creation (this might fail even offline)
    sweep_config = {
        'method': 'random',
        'parameters': {
            'lr': {'values': [0.001, 0.01]}
        }
    }
    sweep_id = wandb.sweep(sweep_config)
    print("✓ wandb.sweep() works offline")
    print(f"Sweep ID: {sweep_id}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    print("This confirms the issue is with sweep creation")

wandb.finish()
