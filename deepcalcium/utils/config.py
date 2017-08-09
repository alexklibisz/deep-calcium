from platform import system
import json
import os


def get_config():
    """Returns config file contents. Makes the file if it doesn't exist."""
    user_home = os.path.expanduser('~')
    base_dir_windows = '%s/Documents/deep-calcium' % os.path.expanduser('~')
    base_dir_unix = '%s/.deep-calcium' % os.path.expanduser('~')
    base_dir = base_dir_windows if system() == 'Windows' else base_dir_unix
    config_path = '%s/deep-calcium.json' % base_dir

    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    if not os.path.exists(config_path):
        config = {
            'datasets_dir': '%s/datasets' % base_dir,
            'checkpoints_dir': '%s/checkpoints' % base_dir
        }
        fp = open(config_path, 'w')
        json.dump(config, fp)
        fp.close()
    else:
        fp = open(config_path, 'r')
        config = json.load(fp)
        fp.close()

    os.makedirs(config['datasets_dir'], exist_ok=True)
    os.makedirs(config['checkpoints_dir'], exist_ok=True)
    return config


# Export config variables.
config = get_config()
DATASETS_DIR = config['datasets_dir']
CHECKPOINTS_DIR = config['checkpoints_dir']
