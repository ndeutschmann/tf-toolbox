from tensorboard.plugins.hparams import api as hp
from tensorflow_core.python.keras.api._v2 import keras as keras

from tf_toolbox.training.model_manager import ModelManager


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

    def create_model(self,*, optimizer_object, loss_function, n_layers, layer_size, layer_activation="relu", reg=0., **opts):

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
            optimizer=optimizer_object,
            loss=loss_function,
            metrics=["accuracy"]
        )

    @property
    def model(self):
        if self._model is not None:
            return self._model
        else:
            raise AttributeError("No model was instantiated")

    def train_model(self, X, y, *, batch_size, epochs, logdir=None, hparams=None, **fit_options):
        callbacks = []
        if logdir is not None:
            metrics_callback = keras.callbacks.TensorBoard(log_dir=logdir)
            callbacks = [metrics_callback]
            if hparams is not None:
                keras_callback=hp.KerasCallback(logdir,hparams)
                callbacks.append(keras_callback)

        return self.model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks, **fit_options)