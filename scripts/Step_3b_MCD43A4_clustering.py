from __future__ import annotations

import argparse

from jpdfusion.clustering import cluster_source
from jpdfusion.config import load_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCD43A4 temporal clustering")
    parser.add_argument("--config", default="config.example.yml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(cluster_source(load_config(args.config), "MCD43A4", args.overwrite))

