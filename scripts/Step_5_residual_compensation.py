from __future__ import annotations

import argparse

from jpdfusion.config import load_config
from jpdfusion.residual import residual_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Residual compensation for final JPDFusion products")
    parser.add_argument("--config", default="config.example.yml")
    parser.add_argument("--sources", nargs="+", choices=("SAR", "MCD43A4"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    print(residual_all(config, args.sources, args.overwrite))

