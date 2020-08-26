"""
Author: Dor Meir
Date: 26.8.2020
For any questions, please contact me at: Dor.meir999@gmail.com
"""

import importlib
import yaml
import utils
from utils import import_process_cluster_output_log
importlib.reload(utils)


def main():
    """
    Run the entire pipeline, print outputs only to a log file.
    All parameters  are configured in configs.yaml.
    :return:
         1. The original data frame with the cluster labeling, with one row for each unique search query.
         2. The aggregated clusters data, with one row for each cluster in each day.
         3. The vectorizors used for representing the search queries
         4. The kmeans models used for clustering the search queries
         ----- For data of more than one day -----
         5. The evaluated features over each cluster and over all days, with two scores of the clusters.
         6. The plots of the some top clusters data on each evaluated feature.
    """
    import_process_cluster_output_log()


if __name__ == "__main__":
    main()
