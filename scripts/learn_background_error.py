import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml

from assimilation_framework.utils.logger import get_logger
from assimilation_framework.utils.misc import load_config, set_random_seed, init_distributed_mode, get_rank
from assimilation_framework.bgerr_learning.runner import LearningBgRunner

def main(args):
    if args.world_size > 1:
        init_distributed_mode(args)
    else:
        args.rank = 0
        args.distributed = False
        args.local_rank = 0
        torch.cuda.set_device(args.local_rank)

    cfg = load_config(args.config)
    cfg["prefix"] = args.prefix
    cfg["date"] = args.date

    seed = cfg.get("random_seed", 42)
    set_random_seed(seed)

    run_dir = os.path.join("experiments", "learning_background", cfg["prefix"])
    os.makedirs(run_dir, exist_ok=True)

    logger = get_logger("learn bg", run_dir, distributed_rank=get_rank(), filename="learn_bg.log", resume=True)
 
    runner = LearningBgRunner(args=args, config=cfg, run_dir=run_dir, logger=logger)
    runner.run()

    logger.info("Learning background completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file.")
    parser.add_argument("--prefix", type=str, required=True, help="Prefix.")
    parser.add_argument("--date", type=str, required=True, help="Date of running experimets.")
    parser.add_argument("--init_method", type=str, default="tcp://127.0.0.1:23456", help="multi process init method")
    parser.add_argument("--world_size", type = int, default=1, help='Number of progress')
    parser.add_argument("--per_cpus", type = int, default=1, help='Number of perCPUs to use')
 
    args = parser.parse_args()
    main(args)
