from __future__ import annotations

import argparse

from jpdfusion.config import load_config
from jpdfusion.copula import copula_source


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit SAR-HLS Gaussian copulas")
    parser.add_argument("--config", default="config.example.yml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(copula_source(load_config(args.config), "SAR", args.overwrite))

