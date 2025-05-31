from petrel_client.client import Client

from assimilation_framework.bgerr_learning.learning_algorithms import GenBeAgent, VaeBeAgent

class LearningBgRunner:
    def __init__(self, args, config, run_dir, logger):
        self.args       = args
        self.config     = config
        self.run_dir    = run_dir
        self.logger     = logger
        self.client     = Client(conf_path="~/petreloss.conf")

    def run(self):
        if self.config["type"] == "genbe":
            model = GenBeAgent(client=self.client, error_path=self.config["error_sample_path"], output_path=self.config["output_path"], lat=self.config["lat"], lon=self.config["lon"])
            model.run()
        elif self.config["type"] == "vae":
            model = VaeBeAgent(client=self.client, logger=self.logger, args=self.args, config=self.config, run_dir=self.run_dir)
            model.run()
        else:
            raise NotImplementedError