import os

import yaml
from tensorboard.plugins.hparams import api as hp
from tensorflow import keras

from tf_toolbox.training import abstract_managers as AM
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


class StandardModelManager(AM.ModelManager):
    """Standard API realization for a model manager"""

    @property
    def hparam(self):
        return self._hparam

    @property
    def metrics(self):
        return self._metrics

    @property
    def model(self):
        if self._model is not None:
            return self._model
        else:
            raise AttributeError("No model was instantiated")

    @model.deleter
    def model(self):
        if self._model is not None:
            del self._model
            self._model = None
        else:
            raise AttributeError("No model was instantiated")

    def save_weights(self, *, logdir, prefix=""):
        """Save the current weights"""
        filename = os.path.join(logdir, "model_info", prefix, "weights.h5")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.model.save_weights(filename)

    def save_hparams(self, *, hparam, logdir, prefix=""):
        """Save the hyperparameters that were used to instantiate and train this model"""
        param_name_dict = dict([(h.name,val) for h,val in hparam.items()])
        filename = os.path.join(logdir, "model_info", prefix, "hparams.yaml")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w+") as hparams_yaml:
            yaml.safe_dump(param_name_dict,stream=hparams_yaml)

    def save_hparams_and_weights(self,*, hparam, logdir, prefix=""):
        """Save the hyperparameters and the current weights"""
        self.save_weights(logdir=logdir, prefix=prefix)
        self.save_hparams(hparam=hparam,logdir=logdir, prefix=prefix)

    def load_weights(self,weight_file_path):
        """Load saved weights into an existing model"""
        self.model.load_weights(weight_file_path)

    def load_weights_from_checkpoint(self, checkpoint_path):
        """Load saved weights from a checkpoint directory into an existing model"""
        filename = os.path.join(checkpoint_path,"weights.h5")
        self.load_weights(filename)

    def create_model_from_hparams(self, hparams_yaml_path, *, optimizer_object):
        """Create a model from a YAML hyperparameter file and an optimizer"""
        with open(hparams_yaml_path, "r") as hparams_yaml:
            hparams = yaml.load(hparams_yaml, Loader=yaml.FullLoader)
        self.create_model(optimizer_object=optimizer_object, **hparams)

    def create_model_from_checkpoint(self ,checkpoint_path, *, optimizer_object):
        """Create a model from the hyperparameters of a checkpoint and an optimizer"""
        hparams_yaml_path = os.path.join(checkpoint_path, "hparams.yaml")
        self.create_model_from_hparams(hparams_yaml_path, optimizer_object=optimizer_object)

    def load_model_from_checkpoint(self, checkpoint_path, *, optimizer_object):
        """Create and load a pre-trained model from a checkpoint"""
        self.create_model_from_checkpoint(checkpoint_path,optimizer_object=optimizer_object)
        self.load_weights_from_checkpoint(checkpoint_path)


class InversibleModelManager(StandardModelManager):
    """Standard API realization for an inversible model manager"""
    @property
    def inverse_model(self):
        if self._inverse_model is not None:
            return self._inverse_model
        else:
            raise AttributeError("No inverse model was instantiated")