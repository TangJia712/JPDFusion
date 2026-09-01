from __future__ import annotations

import argparse

from jpdfusion.config import load_config
from jpdfusion.validation import validate_inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate JPDFusion example inputs")
    parser.add_argument("--config", default="config.example.yml")
    parser.add_argument("--sources", nargs="+", choices=("SAR", "MCD43A4"))
    args = parser.parse_args()
    config = load_config(args.config)
    print(validate_inputs(config, args.sources))

