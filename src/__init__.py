import yaml

def load_configs(path):
    with open(path, 'r') as f:
        configs = yaml.full_load(f)
    return configs

def name_to_class(name):
    return ''.join(n.capitalize() for n in name.split('_'))