from abc import ABC,abstractmethod


class ModelManager(ABC):
    """An abstract class for Managers that create a model with a specific architecture and parameters and register
    them as HParam and Metric objects"""

    @abstractmethod
    def __init__(self, **hp_parameters):
        """Instantiate a manager object with meta-parameters. Meta-parameters are the parameters of the Hparams such
        as ranges for continuous variables"""
        pass

    @property
    @abstractmethod
    def hparam(self):
        """A dictionnary with string keys and HParam values.
        These are to be passed along to an experiment for initialization"""
        pass

    @property
    @abstractmethod
    def metrics(self):
        """A dictionnary with string keys and Metric values.
        These are to be passed along to an experiment for initialization"""
        pass

    @abstractmethod
    def create_model(self, **model_parameters):
        """Instantiate a model with a specific optimizer and loss
         and set of model hyperparameter values"""
        pass

    @property
    @abstractmethod
    def model(self):
        """The currently instantiated model if it exists"""
        pass

    @model.deleter
    @abstractmethod
    def model(self):
        """Delete the current instantiated model"""

    @abstractmethod
    def train_model(self,**training_parameters):
        """Train a model using a given set of training parameters.
        Depending on the model these training parameters might be data and labels, training strategy specifics,
        loss functions, etc.
        """

