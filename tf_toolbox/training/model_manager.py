from abc import ABC,abstractmethod

class ModelManager(ABC):
    """An abstract class for Managers that create a model with a specific architecture and parameters and register
    them as HParam and Metric objects"""

    @abstractmethod
    def __init__(self, **hp_parameters):
        """Instantiate a manager object with meta-parameters. Meta-parameters are the parameters of the Hparams such
        as ranges for continuous variables"""
        pass

    @abstractmethod
    def get_hparam_metric(self):
        """Return two dictionnaries:
            hparams with string keys and HParam values
            metrics with string keys and Metric values
        These are to be passed along to an experiment for initialization"""
        pass

    @abstractmethod
    def create_model(self, **model_parameters):
        """Instantiate a model with a specific set of parameter values"""
        pass

    @abstractmethod
    def train_model(self,*,model,optimizer,**training_parameters):
        """Train a model with an optimizer using a given set of training parameters.
        Depending on the model these training parameters might be data and labels, training strategy specifics,
        loss functions, etc.
        """