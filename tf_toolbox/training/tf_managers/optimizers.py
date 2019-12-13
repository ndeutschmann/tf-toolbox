from tensorboard.plugins.hparams import api as hp
from tensorflow_core.python.keras.optimizer_v2.adam import Adam
from tensorflow_core.python.keras.optimizer_v2.adamax import Adamax
from tensorflow_core.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow_core.python.keras.optimizer_v2.rmsprop import RMSprop

from tf_toolbox.training.optimizer_manager import StandardManager


class SGDManager(StandardManager):
    """A Manager class for the vanilla SGD optimizer"""

    def __init__(self, lr_range = (1.e-8,1.)):
        """Create a SGD manager with a specific learning rate range"""
        super(SGDManager, self).__init__()
        self._hparam = {
            "optimizer": hp.HParam("optimizer",
                                       domain=hp.Discrete(['SGD']),
                                       display_name="Optimizer"),
            "learning_rate": hp.HParam("learning_rate",
                                       domain=hp.RealInterval(*lr_range),
                                       display_name="Learning rate"),
        }

    def create_optimizer(self, learning_rate=0.01, **opts):
        self._optimizer = SGD(learning_rate=learning_rate)


class SGDMomentumManager(SGDManager):
    """A Manager class for the momentum(possibly Nesterov) SGD optimizer"""

    # How to map the hyperparameter label to an argument of the keras SGD constructor
    nesterov_map = {
        "standard": False,
        "nesterov": True
    }

    def __init__(self, lr_range = (1.e-8,1.), momentum_range = (0.,1.)):
        super(SGDMomentumManager, self).__init__(lr_range=lr_range)
        self._hparam['momentum'] = hp.HParam('momentum',description="Momentum Rate",domain=hp.RealInterval(*momentum_range))
        self._hparam['nesterov'] = hp.HParam('nesterov',description="Momentum",domain=hp.Discrete(['standard','nesterov']))

    def create_optimizer(self, learning_rate=0.01, momentum=0., nesterov='standard', **opts):
        self._optimizer = SGD(learning_rate=learning_rate, momentum=momentum, nesterov=self.nesterov_map[nesterov])


class AdamManager(StandardManager):
    """An optimizer for the Adam/Amsgrad/Adamax optimizer"""

    def __init__(self,
                 lr_range = (1.e-8,1.),
                 beta_1_range = (1. - 1.e-1, 1. - 1.e-4),
                 beta_2_range = (1. - 1.e-3, 1. - 1e-6),
                 epsilon_range = (1.e-8,1.e-3)):
        """Create an Adam/Adamax manager with a specific range for hyperparameters
        A last non-tunable hyperparameter is `optimizer`, which can be Adam, Amsgrad, Adamax.

        Args:
            lr_range ():
            beta_1_range ():
            beta_2_range ():
            epsilon_range ():
        """
        super(AdamManager, self).__init__()
        self._hparam = {
            "optimizer": hp.HParam("optimizer",
                                       domain=hp.Discrete(['Adam','Amsgrad','Adamax']),
                                       display_name="Optimizer"),
            "learning_rate": hp.HParam("learning_rate",
                                       domain=hp.RealInterval(*lr_range),
                                       display_name="Learning rate"),
            "beta_1": hp.HParam("beta_1",
                                       domain=hp.RealInterval(*beta_1_range),
                                       display_name="Beta1"),
            "beta_2": hp.HParam("beta_2",
                                       domain=hp.RealInterval(*beta_2_range),
                                       display_name="Beta2"),
            "epsilon": hp.HParam("epsilon",
                                       domain=hp.RealInterval(*epsilon_range),
                                       display_name="Epsilon"),
        }

    def create_optimizer(self,
                         optimizer="Adam",
                         learning_rate=0.01,
                         beta_1=0.9,
                         beta_2=0.999,
                         epsilon=1.e-7,
                         **opts):
        """

        Args:
            optimizer ():
            learning_rate ():
            beta_1 ():
            beta_2 ():
            epsilon ():
            **opts ():

        Returns:

        """

        assert isinstance(optimizer, str)

        if optimizer == "Adam":
            self._optimizer = Adam(learning_rate=learning_rate,
                                   beta_1=beta_1,
                                   beta_2=beta_2,
                                   epsilon=epsilon,
                                   amsgrad=False)
        elif optimizer == "Amsgrad":
            self._optimizer = Adam(learning_rate=learning_rate,
                                   beta_1=beta_1,
                                   beta_2=beta_2,
                                   epsilon=epsilon,
                                   amsgrad=True)
        elif optimizer == "Adamax":
            self._optimizer = Adamax(learning_rate=learning_rate,
                                   beta_1=beta_1,
                                   beta_2=beta_2,
                                   epsilon=epsilon)

        else:
            raise ValueError("Unknown optimizer mode for AdamXManager: {}".format(optimizer))


class RMSpropManager(StandardManager):
    """Manager for the RMSprop optimizer"""

    def __init__(self, lr_range=(1.e-8,1.e8), rho_range = (0.,1.), momentum_range = (0., 1.), epsilon_range = (0.,1.) ):
        super(RMSpropManager, self).__init__()
        self._hparam = {
            "optimizer": hp.HParam("optimizer",
                                   domain=hp.Discrete(['RMSProp']),
                                   display_name="Optimizer"),
            "learning_rate": hp.HParam("learning_rate",
                                       domain=hp.RealInterval(*lr_range),
                                       display_name="Learning rate"),
            "rho": hp.HParam("rho",
                                       domain=hp.RealInterval(*rho_range),
                                       display_name="Rho"),
            "momentum": hp.HParam("momentum",
                                       domain=hp.RealInterval(*momentum_range),
                                       display_name="Momentum rate"),
            "epsilon": hp.HParam("epsilon",
                                  domain=hp.RealInterval(*epsilon_range),
                                  display_name="Epsilon"),
        }

    def create_optimizer(self,
                         learning_rate=0.001,
                         rho=0.9,
                         momentum=0.0,
                         epsilon=1e-07,
                         **opts):

        self._optimizer = RMSprop(learning_rate=learning_rate,rho=rho,momentum=momentum,epsilon=epsilon)