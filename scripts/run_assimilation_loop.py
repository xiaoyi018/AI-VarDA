import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml

from assimilation_framework.utils.logger import get_logger
from assimilation_framework.utils.misc import load_config, set_random_seed
from assimilation_framework.assimilation_core.runner import AssimilationRunner

def main(args):
    cfg = load_config(args.config)
    cfg["prefix"] = args.prefix
    cfg["date"] = args.date

    seed = cfg.get("random_seed", 42)
    set_random_seed(seed)

    run_dir = os.path.join("experiments", "assimilation", cfg["prefix"])
    os.makedirs(run_dir, exist_ok=True)

    logger = get_logger("run da", run_dir, filename="run_da.log", resume=cfg["resume"])
 
    runner = AssimilationRunner(config=cfg, run_dir=run_dir, logger=logger)
    runner.run()

    logger.info("Assimilation completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file.")
    parser.add_argument("--prefix", type=str, required=True, help="Prefix.")
    parser.add_argument("--date", type=str, required=True, help="Date of running experimets.")
    args = parser.parse_args()
    main(args)
