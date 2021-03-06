from .model_manager import ModelManager
from .optimizer_manager import OptimizerManager
from time import time
from abc import abstractmethod
import tensorboard.plugins.hparams.api as hp
import os

class ExperimentManager:
    """A manager class for overall experiments
    This class provides the higher level interface to models and optimizers and is supposed to serve as framework
    to make training and scans smooth across different settings."""
    def __init__(self,model_manager:ModelManager,optimizer_manager:OptimizerManager,*,logdir,run_name_template="run", log_names=True):
        """

        Args:
            model_manager ():
            optimizer_manager ():
            logdir ():
            run_name_template ():
        """
        self.model_manager = model_manager
        self.optimizer_manager = optimizer_manager

        self.hp_dict = dict(model_manager.hparam, **optimizer_manager.hparam)

        # Logging the run name makes it easier to match the Scalars page to the HParams page
        # However it can overcrowd the Hparams page when repeating the same experiment so optionally turn it off.
        self.log_names = log_names
        if log_names:
            self.hp_dict.update({"run_name": hp.HParam('run_name',display_name="Run name")})

        self.run_name = None
        self.logdir = logdir
        self.run_name_template = run_name_template
        self.run_id = 0
        self.epoch = 0


    def hp_dict_template(self):
        """Output a string that describes a dictionnary { hyperparameter1: X1, ... } with empty X1,
         which can be filled in by the user for values of Xi and fed into the run methods"""
        dict_str = """{\n"""
        for k in self.hp_dict.keys():
            dict_str += """{}: ,\n""".format(k)
        dict_str += "}"
        return dict_str

    def prepare_run(self,run_name=None, run_id=None, use_timestamp=True, **create_opts):
        """

        Args:
            run_name ():
            run_id ():
            use_timestamp ():
            **create_opts ():

        Returns:

        """

        if self.run_name is not None:
            raise RuntimeError("Please terminate the current run before starting a new one")

        self.optimizer_manager.create_optimizer(**create_opts)
        self.model_manager.create_model(optimizer_object=self.optimizer_manager.optimizer,**create_opts)

        _full_run_name = "{run_name}_{run_id}{timestamp}".format(
            run_name=run_name or self.run_name_template,
            run_id=run_id or self.run_id,
            timestamp="_"+str(time() if use_timestamp else "")
        )

        self.run_id = run_id or self.run_id
        self.run_name = _full_run_name
        self.epoch = 0

    @abstractmethod
    def start_model_manager_training(self,**opts):
        """Experiment-type specific command

        Comment on the API:
        This is an API-defining function: sub_classes implementing the lower-level `start_model_manager_training`
        are supposed to implement it all in the same spirit that this low-level method should take all its arguments
        explicitly from the do_run call in this method

        It is ok to implement a new do_run in a subclass, (see LoggingExperimentManager below, which fills in info from
        the internal config that is saved) but it should always call super to handle the lower-level call to
        start_model_manager_training.
        """
        pass

    def do_run(self,**run_opts):
        """Start a training run
        Arguments depend on the specific model/training mode. Look at the signature of self.model_manager.train_model
        for further details.

        Comment on the API:
        This is an API-defining function: sub_classes implementing the lower-level `start_model_manager_training`
        are supposed to implement it all in the same spirit that this low-level method should take all its arguments
        explicitly from the do_run call in this method

        It is ok to implement a new do_run in a subclass, (see LoggingExperimentManager below, which fills in info from
        the internal config that is saved) but it should always call super to handle the lower-level call to
        start_model_manager_training.
        """
        if self.run_name is None:
            raise RuntimeError("No run initialized")

        # Logging the run name as a Hparam if the options was set to True (see __init__ for rationale)
        # TODO This is not a nice implementation of run name storing.
        assert "run_name" not in run_opts, "run_name is a reserved keyword. Run names are managed internally"
        if self.log_names:
            run_opts["run_name"] = self.run_name

        print("Starting run "+self.run_name)
        # The main reason for this class is this line below:
        # build a Hparam-keyed dictionnary for proper logging
        hparam = dict([(self.hp_dict[k],run_opts[k]) for k in self.hp_dict])
        logdir = os.path.join(self.logdir,self.run_name)

        result = self.start_model_manager_training(logdir=logdir, hparam=hparam, epoch_start=self.epoch, **run_opts)

        if "epochs" in run_opts:
            self.epoch+=run_opts['epochs']
        return result

    def end_run(self):
        self.run_name = None
        self.run_id += 1
        self.epoch = 0
        del self.model_manager.model


class LoggingExperimentManager(ExperimentManager):
    """Generic Experiment manager that calls the model manager's `train_model` method with a list of logger functions
    that will be called on metrics. The logger must have the same signature as tf.summary.scalar
    """
    # TODO:
    # The goal is to make this class a user-facing class whose definition is a main element of the model manager API
    # Ideally, one would define an abstract ModelManager subclass that has all the abstract methods with the right
    # signatures to enforce this structure. For now this is essentially enforced by us using only the FlowManager class
    # which implements one unique training method for all our architectures, making the propagation of changes
    # manageable even without a clear interface requirement between models and experiements.

    def __init__(self,model_manager:ModelManager,optimizer_manager:OptimizerManager,*,logdir,run_name_template="run"):
        super(LoggingExperimentManager, self).__init__(model_manager, optimizer_manager, logdir=logdir,
                                                       run_name_template=run_name_template, log_names=True)
        self.run_opts = None

    def prepare_run(self, **opts):
        super(LoggingExperimentManager, self).prepare_run(run_name=None, run_id=None, use_timestamp=True, **opts)
        self.run_opts = opts

    def do_run(self,**runtime_options):
        # TODO:
        # for now the distinction between the options provided in prepare run and runtime_options is purely semantic.
        # There are no check as to which is which and this is a bad thing.
        # The goal is to be able to wrap this whole thing in a Sacred experiment, in which one would like to log the
        # options of prepare_run as a experiment.config while the runtime options are things like data,
        # which we might not want to log as part of the configuration. For now, the separation is left to the user,
        # which is of course dangerous.
        if self.run_opts is None:
            raise RuntimeError("Prepare a run before starting it")
        return super(LoggingExperimentManager, self).do_run(**self.run_opts, **runtime_options)

    def end_run(self):
        super(LoggingExperimentManager, self).end_run()
        self.run_opts = None

    def start_model_manager_training(self, logger_functions=(), epoch_start=0, *, logdir, hparam, **runtime_options):
        # TODO runtime options ambiguity issue, see comment in do_run
        return self.model_manager.train_model(epoch_start=self.epoch,
                                              logdir=logdir,
                                              hparam=hparam,
                                              logger_functions=logger_functions,
                                              **runtime_options)
