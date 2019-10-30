from abc import ABC,abstractmethod,abstractproperty
from tensorflow.keras.optimizers import SGD

class OptimizerManager(ABC):
    """An abstract class of Managers that create an optimizer with a specific set of options.
    These options are cast into HParam objects.
    """


    @abstractmethod
    def __init__(self, **hp_parameters):
        """Instantiate a manager object with meta-parameters. Meta-parameters are the parameters of the Hparams such
        as ranges for continuous variables"""

    @abstractmethod
    def get_hparam(self):
        """Return a dictionnary with string keys and HParam values.
        These are to be passed along to an experiment for initialization"""
        pass

    @abstractmethod
    def create_optimizer(self, **optimizer_parameter):
        """Instantiate an optimizer with a specific set of parameter values"""
        pass