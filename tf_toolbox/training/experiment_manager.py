from .model_manager import ModelManager
from .optimizer_manager import OptimizerManager
from time import time
from abc import abstractmethod
import tensorboard.plugins.hparams.api as hp
import tensorflow as tf
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
        pass

    def do_run(self,**run_opts):
        """Start a training run
        Arguments depend on the specific model/training mode. Look at the signature of self.model_manager.train_model
        for further details.
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
        hparam_values = dict([(self.hp_dict[k],run_opts[k]) for k in self.hp_dict])
        run_logdir = os.path.join(self.logdir,self.run_name)

        result = self.start_model_manager_training(run_logdir=run_logdir,hparam_values=hparam_values,**run_opts)

        if "epochs" in run_opts:
            self.epoch+=run_opts['epochs']
        return result

    def end_run(self):
        self.run_name = None
        self.run_id += 1
        self.epoch = 0
        del self.model_manager.model


class TBExperimentManager(ExperimentManager):

    def setup_tb(self):
        with tf.summary.create_file_writer(self.logdir).as_default():
            hp.hparams_config(hparams=self.hp_dict.values(),metrics=self.model_manager.metrics.values())

    def start_model_manager_training(self,*,run_logdir,hparam_values,**run_opts):
        with tf.summary.create_file_writer(run_logdir).as_default():
            hp.hparams(hparam_values)
            result = self.model_manager.train_model(logdir=run_logdir, epoch_start=self.epoch, hparam=hparam_values, logger_functions=[tf.summary.scalar], **run_opts)