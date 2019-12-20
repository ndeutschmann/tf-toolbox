from tensorboard.plugins.hparams import api as hp
from tensorflow import summary

from tf_toolbox.training.experiment_manager import LoggingExperimentManager


class TBExperimentManager(LoggingExperimentManager):
    def setup_tb(self):
        with summary.create_file_writer(self.logdir).as_default():
            hp.hparams_config(hparams=self.hp_dict.values(),metrics=self.model_manager.metrics.values())

    def start_model_manager_training(self, epoch_start=0, *, logdir, hparam, other_loggers=[], **runtime_options):

        assert hasattr(self.model_manager,"save_hparams")
        self.model_manager.save_hparams(hparam=hparam, logdir=logdir)

        with summary.create_file_writer(logdir).as_default():
            hp.hparams(hparam)
            # TODO:
            # For now the call to the training function of LoggingExperimentManager is quite superfluous.
            # However the idea is to have LoggingExperimentManager specify one of the user-facing API with detailed
            # specifications on requirements.
            return super(TBExperimentManager, self).\
                start_model_manager_training(logdir=logdir,
                                             hparam=hparam,
                                             epoch_start=epoch_start,
                                             logger_functions=[summary.scalar]+other_loggers,
                                             **runtime_options)