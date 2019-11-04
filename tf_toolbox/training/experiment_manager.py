from .abstract_managers import ModelManager, OptimizerManager
from time import time
import tensorboard.plugins.hparams.api as hp

class ExperimentManager:
    """A manager class for overall experiments
    This class provides the higher level interface to models and optimizers and is supposed to serve as framework
    to make training and scans smooth across different settings.
    """
    def __init__(self,model_manager:ModelManager,optimizer_manager:OptimizerManager,*,logdir,run_name_template="run_"):
        self.model_manager = model_manager
        self.optimizer_manager = optimizer_manager

        self.hp_dict = dict(model_manager.hparam, **optimizer_manager.hparam)
        self.run_name = None
        self.logdir = logdir
        self.run_name_template = run_name_template
        self.run_id = 0
        # TODO  Have a full default behavior based on logdir/run_name_template_{id}_{timestamp}
        # TODO  and scan directory to initiate the id at (last value)+1
        # TODO  Alternative: save setup in logdir? pickle/yaml?

    def setup_tb(self):
        hp.hparams_config(hparams=self.hp_dict,metrics=self.model_manager.metrics)

    def prepare_run(self,run_name=None, run_id=None, use_timestamp=True, **create_opts):
        self.optimizer_manager.create_optimizer(**create_opts)
        self.model_manager.create_model(optimizer=self.optimizer_manager.optimizer,**create_opts)

        _full_run_name = "{run_name}_{run_id}{timestamp}".format(
            run_name=run_name or self.run_name_template,
            run_id=run_id or self.run_id,
            timestamp="_"+str(time.time() if use_timestamp else "")
        )

        self.run_id = run_id
        self.run_name = _full_run_name

    def do_run(self,**run_opts):
        """Start a training run
        Arguments depend on the specific model/training mode. Look at the signature of self.model_manager.train_model
        for further details.
        """
        if self.run_name is None:
            raise RuntimeError("No run initialized")

        # The main reason for this class is this line below:
        # build a Hparam-keyed dictionnary for proper logging
        hparams_values = dict([(self.hp_dict[k],run_opts[k]) for k in self.hp_dict])
        self.model_manager.train_model(logdir=self.logdir+"/"+self.run_name,hparams=hparams_values,**run_opts)

    def end_run(self):
        self.run_name = None
        self.run_id+=1
