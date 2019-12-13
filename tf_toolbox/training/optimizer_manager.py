from abc import ABC,abstractmethod


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

class StandardManager(OptimizerManager):
    """Second layer of meta class that implements generic access to optimizer and hparam"""

    def __init__(self):
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


