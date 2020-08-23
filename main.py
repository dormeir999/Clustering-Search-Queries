

import importlib
import yaml
import utils

importlib.reload(utils)

from utils import import_process_cluster_output_log

with open('configs.yaml') as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)


def main():
    import_process_cluster_output_log()

if __name__ == "__main__":
    main()
