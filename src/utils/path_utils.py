from datetime import datetime

def get_timestamp():
    """Get current timestamp for model saving"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_date_folder():
    """Get folder name based on current date"""
    return datetime.now().strftime("%Y%m%d")

def setup_model_paths(trainer_params, model_name):
    """Setup model saving paths and filenames"""
    if not trainer_params.get('save_model_folders'):
        trainer_params['save_model_folders'] = [
            get_date_folder(),
            model_name
        ]
    
    if not trainer_params.get('save_model_filename'):
        trainer_params['save_model_filename'] = get_timestamp()
    
    return trainer_params 