import datetime
import os
import importlib
import shutil

def get_exp_dir(expname):
    current_time = datetime.datetime.now()
    current_time_str = current_time.strftime(f"{expname}_%m%d_%H%M%S")
    root_dir = os.getcwd()
    root_dir_log = os.path.join(root_dir, "mc_to_sky/logs")
    if not os.path.exists(root_dir_log):
        os.mkdir(root_dir_log)

    exp_dir = os.path.join(root_dir_log, current_time_str)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    return exp_dir

def check_and_mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def build_model(hypes, return_cls=False):
    model_args = hypes['model']

    model_name = model_args['name']
    model_filename = 'mc_to_sky.model.' + model_name
    model_lib = importlib.import_module(model_filename)
    model_cls = None
    target_model_name = model_name.replace('_', '')

    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model_cls = cls
    if return_cls:
        return model_cls

    model = model_cls(hypes) # note it should be the full hypes
    
    return model

def restore_training(ckpt_path):
    """
    Restore training from a checkpoint.

    >>> ckpt.keys()
    >>> dict_keys(['epoch', 'global_step', 'pytorch-lightning_version', 'state_dict', 'loops', 'callbacks', 'optimizer_states', 'lr_schedulers', 'hparams_name', 'hyper_parameters'])

    Args:
        ckpt_path : str
            path to checkpoint file that end with .ckpt
    
    """
    pass


def backup_script(full_path, folders_to_save=["model", "data_utils", "utils", "loss", "tools"]):
    target_folder = os.path.join(full_path, 'scripts')
    if not os.path.exists(target_folder):
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)
    
    current_path = os.path.dirname(__file__)  # __file__ refer to this file, then the dirname is "?/tools"

    for folder_name in folders_to_save:
        ttarget_folder = os.path.join(target_folder, folder_name)
        source_folder = os.path.join(current_path, f'../{folder_name}')
        shutil.copytree(source_folder, ttarget_folder, dirs_exist_ok=True)