import os
import yaml


def get_config(path):
    out = {}
    if os.path.exists(path):
        with open(path, 'r') as f:
            out = yaml.safe_load(f)

    return out
