"""Command-line orchestration for the complete JPDFusion workflow."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .clustering import cluster_all
from .config import load_config
from .copula import copula_all
from .preprocessing import preprocess_all
from .residual import residual_all
from .validation import validate_inputs

STEPS = ("validate", "preprocess", "cluster", "copula", "residual")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jpdfusion",
        description="Run the reproducible single-band HLS JPDFusion pipeline.",
    )
    parser.add_argument("--config", default="config.example.yml", help="YAML configuration file")
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS), help="Stages to run in order")
    parser.add_argument("--sources", nargs="+", choices=("SAR", "MCD43A4"), help="Override config sources")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing outputs")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    cfg = load_config(args.config)
    sources = args.sources or list(cfg["sources"])
    Path(cfg["paths"]["outputs"]).mkdir(parents=True, exist_ok=True)
    logging.getLogger(__name__).info("Project root: %s", cfg["project_root"])
    logging.getLogger(__name__).info("Sources: %s", ", ".join(sources))

    actions = {
        "validate": lambda: validate_inputs(cfg, sources),
        "preprocess": lambda: preprocess_all(cfg, sources, overwrite=args.overwrite),
        "cluster": lambda: cluster_all(cfg, sources, overwrite=args.overwrite),
        "copula": lambda: copula_all(cfg, sources, overwrite=args.overwrite),
        "residual": lambda: residual_all(cfg, sources, overwrite=args.overwrite),
    }
    for step in args.steps:
        logging.getLogger(__name__).info("Starting stage: %s", step)
        actions[step]()
    logging.getLogger(__name__).info("JPDFusion pipeline completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())

