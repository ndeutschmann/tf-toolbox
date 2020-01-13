import os
from sacred import Experiment
from ..experiment_manager import LoggingExperimentManager
from ..abstract_managers import ModelManager,OptimizerManager

# TODO:
# We want to provide several levels of abstraction.
# Both LoggingExperimentManager and TBExperimentManager can already be wrapped in a user defined script
# as seen in https://gist.github.com/ndeutschmann/b538c3b58cc4ae53a0e0e9df4c25ba85
# This would be enough for running myself, but it will be annoying to redo the same kind of thing for each
# type of experiment.
#
# Level one: very basic janitorial things like
#   - handling observer setting through the experiment manager to make all logging properly organized
#   - ensure consistent definition of the model yaml and hd5 info as artifacts
#   - other stuff
# DONE in BasicSacredExperiment
#
# Level two: fully inclusive package:
#   - treat the sacred experiment as an attribute
#   - generate the @main and @config functions automatically

class BasicSacredExperiment(LoggingExperimentManager):
    """Most basic experiment: take the sacred run object and manipulate it directly
    See https://gist.github.com/ndeutschmann/b538c3b58cc4ae53a0e0e9df4c25ba85 v4 (08.01.2020) for usage
    """
    def start_model_manager_training(self, epoch_start=0, *, run , logdir, hparam, other_loggers=[], score_metric='loss', **runtime_options):

        assert hasattr(self.model_manager,"save_hparams")
        self.model_manager.save_hparams(hparam=hparam, logdir=logdir,prefix="")

        history = super(BasicSacredExperiment, self).\
            start_model_manager_training(logdir=logdir,
                                         hparam=hparam,
                                         epoch_start=epoch_start,
                                         logger_functions=[run.log_scalar]+other_loggers,
                                         **runtime_options)
        run.add_artifact(os.path.join(logdir, "model_info", "hparams.yaml"))
        if "save_best" in runtime_options and runtime_options['save_best']:
            run.add_artifact(os.path.join(logdir, "model_info", "best", "weights.h5"))
        return history.history[score_metric][-1]

class SacredExperiment(BasicSacredExperiment):
    """TODO"""
    def __init__(self,model_manager:ModelManager,optimizer_manager:OptimizerManager,*,logdir,experiment:Experiment):
        super(SacredExperiment, self).__init__(model_manager,
                                               optimizer_manager,
                                               logdir=logdir,
                                               run_name_template=experiment.get_experiment_info()['name'])
        self.sacred_exp = experiment

    def prepare_run(self, **opts):
        super(SacredExperiment, self).prepare_run(**opts)
        self.sacred_exp.add_config(opts)