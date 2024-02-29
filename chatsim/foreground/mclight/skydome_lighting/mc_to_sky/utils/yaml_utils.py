import yaml
import os

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
def dump_yaml(data, savepath):
    with open(os.path.join(savepath, 'config.yaml'), 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)