from abc import ABC,abstractmethod
import tensorboard.plugins.hparams.api as hp
import tensorflow.keras as keras

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
    def create_model(self,*, optimizer, loss, **model_parameters):
        """Instantiate a model with a specific optimizer and loss
         and set of model hyperparameter values"""
        pass

    @property
    @abstractmethod
    def model(self):
        """The currently instantiated model if it exists"""
        pass

    @abstractmethod
    def train_model(self,**training_parameters):
        """Train a model using a given set of training parameters.
        Depending on the model these training parameters might be data and labels, training strategy specifics,
        loss functions, etc.
        """

class DenseRectClassifierManager(ModelManager):
    """A dense rectangular classifier (all layers have the same size)"""
    def __init__(self,*,input_size,n_classes,activations=("relu",),losses=("categorical_crossentropy",)):
        self.input_size = input_size
        self.n_classes = n_classes

        self._hparam = {
            "n_layers": hp.HParam("n_layers",
                                  domain=hp.IntInterval(1,100),
                                  display_name="Depth"
                                  ),
            "layer_size": hp.HParam("layer_size",
                                    domain=hp.IntInterval(1,100),
                                    display_name="Width"),
            "layer_activation": hp.HParam("layer_activation",
                                          domain=hp.Discrete(activations),
                                          display_name="Activation Fct."),
            "reg": hp.HParam("reg",
                             domain=hp.RealInterval(0.,1.),
                             display_name="Reg."),
            "loss_function": hp.HParam("loss_function",
                                       domain=hp.Discrete(losses),
                                       display_name="Loss Fct.")

        }

        self._metrics = {
            "accuracy" : hp.Metric("accuracy",display_name="Accuracy")
        }

        self._model = None

    @property
    def hparam(self):
        return self._hparam

    @property
    def metrics(self):
        return self._metrics

    def create_model(self,*, optimizer, loss, n_layers, layer_size, layer_activation="relu", reg=0.):

        self._model = keras.Sequential()
        self._model.add(keras.Input(shape=(self.input_size,)))
        for i in range(n_layers):
            self._model.add(keras.layers.Dense(layer_size,
                                        activation=layer_activation,
                                        kernel_regularizer=keras.regularizers.l2(reg)
            ))
        self._model.add(keras.layers.Dense(self.n_classes,
                                    activation="softmax",
                                    kernel_regularizer=None))

        self._model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=["accuracy"]
        )

    @property
    def model(self):
        if self._model is not None:
            return self._model
        else:
            raise AttributeError("No model was instantiated")
