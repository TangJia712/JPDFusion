from __future__ import annotations

import argparse

from jpdfusion.config import load_config
from jpdfusion.preprocessing import preprocess_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear interpolation and SG filtering")
    parser.add_argument("--config", default="config.example.yml")
    parser.add_argument("--sources", nargs="+", choices=("SAR", "MCD43A4"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    print(preprocess_all(config, args.sources, args.overwrite))

