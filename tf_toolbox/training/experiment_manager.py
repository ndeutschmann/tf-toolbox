from .model_manager import ModelManager
from .optimizer_manager import OptimizerManager
from time import time
import tensorboard.plugins.hparams.api as hp
import tensorflow as tf

class ExperimentManager:
    """A manager class for overall experiments
    This class provides the higher level interface to models and optimizers and is supposed to serve as framework
    to make training and scans smooth across different settings."""
    def __init__(self,model_manager:ModelManager,optimizer_manager:OptimizerManager,*,logdir,run_name_template="run"):
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
        self.run_name = None
        self.logdir = logdir
        self.run_name_template = run_name_template
        self.run_id = 0
        self.epoch = 0
        # TODO  Have a full default behavior based on logdir/run_name_template_{id}_{timestamp}
        # TODO  and scan directory to initiate the id at (last value)+1
        # TODO  Alternative: save setup in logdir? pickle/yaml?

    def setup_tb(self):
        with tf.summary.create_file_writer(self.logdir).as_default():
            hp.hparams_config(hparams=self.hp_dict.values(),metrics=self.model_manager.metrics.values())

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

    def do_run(self,**run_opts):
        """Start a training run
        Arguments depend on the specific model/training mode. Look at the signature of self.model_manager.train_model
        for further details.
        """
        if self.run_name is None:
            raise RuntimeError("No run initialized")

        print("Starting run "+self.run_name)
        # The main reason for this class is this line below:
        # build a Hparam-keyed dictionnary for proper logging
        hparams_values = dict([(self.hp_dict[k],run_opts[k]) for k in self.hp_dict])

        result = self.model_manager.train_model(logdir=self.logdir+"/"+self.run_name,hparam=hparams_values, epoch_start=self.epoch,**run_opts)
        if "epochs" in run_opts:
            self.epoch+=run_opts['epochs']
        return result

    def end_run(self):
        self.run_name = None
        self.run_id += 1
        self.epoch = 0
        del self.model_manager.model