"""Real Hyrax verbs for batch tree training/inference, with no library patches."""

import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from hyrax.context import update_context  # noqa: E402
from hyrax.verbs.verb_registry import Verb, hyrax_verb  # noqa: E402


@hyrax_verb
class TrainTrees(Verb):
    cli_name = "wdb_train"
    description = "Fit full-table WDB trees with nested threshold selection and resumable folds."

    @staticmethod
    def setup_parser(parser):
        pass

    def run_cli(self, args=None):
        return self.run()

    def run(self, *, minutes=5.0, max_new_fits=None):
        from .config import TrainingConfig
        from .training import train

        # Normalize TOMLKit arrays/scalars before dataclasses or sklearn see them.
        settings = TrainingConfig(**json.loads(json.dumps(dict(self.config["wdb"]))))
        update_context(results_dir=Path(settings.output), verb=self.cli_name, config=self.config)
        result = train(settings, minutes=minutes, max_new_fits=max_new_fits)
        from hyrax.config_utils import log_runtime_config

        log_runtime_config(self.config, Path(settings.output))
        return dict(result, engine="Hyrax", verb=self.cli_name)


@hyrax_verb
class PredictTrees(Verb):
    cli_name = "wdb_predict"
    description = "Score portable WDB features with a completed fold-model bundle."

    @staticmethod
    def setup_parser(parser):
        pass

    def run_cli(self, args=None):
        raise ValueError("Use the wdb-classifier predict CLI or the Python wdb_predict method")

    def run(self, *, run, features, output, budget=0.05, allow_training_overlap=False):
        from .training import predict

        update_context(results_dir=str(run), verb=self.cli_name, config=self.config)
        return predict(run, features, output, budget=budget, allow_training_overlap=allow_training_overlap)


def create_workflow(config=None):
    from hyrax import Hyrax

    workflow = Hyrax(setup_logging=False)
    if config is not None:
        workflow.set_config("wdb", config.as_dict())
    return workflow
