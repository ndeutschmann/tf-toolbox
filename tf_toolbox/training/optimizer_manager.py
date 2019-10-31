from abc import ABC,abstractmethod,abstractproperty
from tensorflow.keras.optimizers import SGD
import tensorboard.plugins.hparams.api as hp

class OptimizerManager(ABC):
    """An abstract class of Managers that create an optimizer with a specific set of options.
    These options are cast into HParam objects.
    """


    @abstractmethod
    def __init__(self, **hp_parameters):
        """Instantiate a manager object with meta-parameters. Meta-parameters are the parameters of the Hparams such
        as ranges for continuous variables"""

    @property
    @abstractmethod
    def hparam(self):
        """A dictionnary with string keys and HParam values.
        These are to be passed along to an experiment for initialization"""
        pass

    @abstractmethod
    def create_optimizer(self, **optimizer_parameter):
        """Instantiate an optimizer with a specific set of parameter values"""
        pass

    @property
    @abstractmethod
    def optimizer(self):
        """The currently instantiated optimizer if it exists"""
        pass


class SGDManager(OptimizerManager):
    """A Manager class for the vanilla SGD optimizer"""

    def __init__(self, lr_range = (1.e-8,1.)):
        """Create a SGD manager with a specific learning rate range"""
        self._hparam = {
            "optimizer": hp.HParam("optimizer",
                                       domain=hp.Discrete(['naive SGD']),
                                       display_name="Optimizer"),
            "learning_rate": hp.HParam("learning_rate",
                                       domain=hp.RealInterval(*lr_range),
                                       display_name="Learning rate"),
        }

        self._optimizer = None

    @property
    def hparam(self):
        return self._hparam

    @property
    def optimizer(self):
        if self._optimizer is not None:
            return self._optimizer
        else:
            raise AttributeError("No optimizer was instantiated")

    def create_optimizer(self, learning_rate=0.01):
        self._optimizer = SGD(learning_rate=learning_rate)
