from tensorboard.plugins.hparams import api as hp
from tensorflow import summary

from tf_toolbox.training.experiment_manager import ExperimentManager


class TBExperimentManager(ExperimentManager):
    def setup_tb(self):
        with summary.create_file_writer(self.logdir).as_default():
            hp.hparams_config(hparams=self.hp_dict.values(),metrics=self.model_manager.metrics.values())

    def start_model_manager_training(self, *, run_logdir, hparam_values, other_loggers=[], **run_opts):

        assert hasattr(self.model_manager,"save_hparams")
        self.model_manager.save_hparams(hparam=hparam_values, logdir=run_logdir)

        with summary.create_file_writer(run_logdir).as_default():
            hp.hparams(hparam_values)
            result = self.model_manager.train_model(logdir=run_logdir, epoch_start=self.epoch, logger_functions=[summary.scalar]+other_loggers, **run_opts)