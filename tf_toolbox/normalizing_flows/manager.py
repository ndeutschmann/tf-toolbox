import tensorflow as tf
import tensorflow.keras as keras
import tensorboard.plugins.hparams.api as hp
import tensorflow_probability as tfp
from tqdm.autonotebook import tqdm
from .layers import AddJacobian,PieceWiseLinear,RollLayer
import tf_toolbox.training.abstract_managers as AM

class RollingPWlinearNormalizingFlow(AM.ModelManager):
    """A manager for normalizing flows with piecewise linear coupling cells interleaved with rolling layers that
    apply cyclic permutations on the variables. All cells have the same number of pass through variables and the
    same step size in the cyclic permutation.
    Each coupling cell has a fully connected NN with a fixed number of layers (depth) of fixed size (width)

    Hyperparameters:
    - n_pass_through
    - n_bins
    - nn_width
    - nn_depth
    - nn_activation
    - roll_step
    - l2_reg
    - dropout_rate
    - TODO training mode
    """
    def __init__(self,
                 *
                 n_flow,
                 n_pass_through_domain = None,
                 n_cells_domain=(1,10),
                 n_bins_domain = (2,10),
                 nn_width_domain = None,
                 nn_depth_domain = (1,10),
                 nn_activation_domain = ("relu",),
                 roll_step_domain = None,
                 l2_reg_domain = (0,1),
                 dropout_rate_domain = (0,1),
                 **init_opts
):
        self.n_flow = n_flow
        self._model = None
        self._inverse_model = None

        if n_pass_through_domain is None:
            _n_pass_through_domain = [1,n_flow-1]
        else: _n_pass_through_domain=n_pass_through_domain

        if nn_width_domain is None:
            _nn_width_domain = [n_bins_domain[0],5*n_bins_domain[1]]
        else: _nn_width_domain=nn_width_domain

        if roll_step_domain is None:
            _roll_step_domain = [1,n_flow-1]
        else: _roll_step_domain=roll_step_domain

        self._hparam = {
            "n_pass_through": hp.HParam("n_pass_through", domain=hp.IntInterval(_n_pass_through_domain),display_name="# Pass"),

            "n_cells": hp.HParam("n_cells",domain=hp.IntInterval(n_cells_domain),display_name="# Cells"),

            "n_bins": hp.HParam("n_bins", domain=hp.IntInterval(n_bins_domain),display_name="# Bins"),

            "nn_width": hp.HParam("nn_width", domain=hp.IntInterval(_nn_width_domain),display_name="NN width"),

            "nn_depth": hp.HParam("nn_depth", domain=hp.IntInterval(nn_depth_domain),display_name="NN depth"),

            "nn_activation": hp.HParam("nn_activation", domain=hp.Discrete(nn_activation_domain),display_name="NN activ. fct."),

            "roll_step": hp.HParam("roll_step", domain=hp.IntInterval(_roll_step_domain),display_name="Roll step"),

            "l2_reg": hp.HParam("l2_reg", domain=hp.RealInterval(l2_reg_domain),display_name="L2 reg."),

            "dropout_rate": hp.HParam("dropout_rate", domain=hp.RealInterval(dropout_rate_domain),display_name="Dropout rate"),

        }

        self._metrics = {
            "std": hp.Metric("std", display_name="Integrand standard deviation")
        }

    @property
    def hparam(self):
        return self._hparam

    @property
    def model(self):
        if self._model is not None:
            return self._model
        else:
            raise AttributeError("No model was instantiated")

    @property
    def inverse_model(self):
        if self._inverse_model is not None:
            return self._inverse_model
        else:
            raise AttributeError("No inverse model was instantiated")

    format_input = AddJacobian()

    def create_model(self,*,
                     n_pass_through,
                     n_cells,
                     n_bins,
                     nn_width,
                     nn_depth,
                     nn_activation="relu",
                     roll_step,
                     l2_reg=0,
                     dropout_rate=0,
                     **opts
                     ):

        self._model = keras.Sequential()

        for i_cell in range(n_cells):
            nn_layers = [nn_width]*nn_depth
            self._model.add(
                PieceWiseLinear(self.n_flow,n_pass_through,n_bins=n_bins,nn_layers=nn_layers,reg=l2_reg,dropout=dropout_rate)
            )
            self._model.add(RollLayer(roll_step))

        self._inverse_model = keras.Sequential([l.inverse for l in reversed(self._model.layers)])

    def train_model(self, train_mode = "variance_forward", **train_opts):
        """Training method that dispatches the model into the different training modes.
        Training modes are implemented as methods named with the convention _train_{train_mode}
        and are expected to return TODO
        Currently implemented modes are
            - variance_forward
                options are (TODO)
        """
        try:
            trainer = getattr(self,"_train_"+train_mode)
        except AttributeError as error:
            raise AttributeError("The train mode %s does not exist."%train_mode)

        return trainer(**train_opts)

    def _train_variance_forward(self, f, n_batch = 10000, n_epochs=10, *, optimizer, logging=True, log_tb=True, pretty_progressbar=True, **train_opts):
        """Train the model using the integrand variance as loss and compute the Jacobian in the forward pass
        (fixed latent space sample mapped to a phase space sample)
        See notes equation TODO
        """

        # Instantiate a pretty launchbar if needed
        if pretty_progressbar:
            epoch_progress = tqdm(range(n_epochs),leave=False,desc="Loss: {0:.3e} | Epoch".format(0.))
        else:
            epoch_progress = range(n_epochs)

        # Keep track of metric history if needed
        if logging:
            history = keras.callbacks.History()

        # Loop over epochs
        for i in epoch_progress:
            with tf.GradientTape() as tape:
                # Output a sample of (phase-space point, forward Jacobian)
                XJ = self.model(                                            # Pass through the model
                    self.format_input(                                      # Append a unit Jacobian to each point
                        tf.random.uniform((n_batch, self.flow_size), 0, 1)  # Generate a batch of points in latent space
                    )
                )
                # Separate the points and their Jacobians:
                # This sample is fixed, we optimize the Jacobian
                X = tf.stop_gradient(XJ[:,:-1])
                # The Jacobian is the last entry of each point
                J = XJ[:,-1]
                # Apply function vlaues
                fX = f(X)

                # The Monte Carlo integrand is fX*J: we minimize its variance
                loss = tf.math.reduce_var(fX*J)
                std = tf.math.sqrt(tf.stop_gradient(loss))
                loss = tf.math.log(loss)
                # Regularization losses are collected as we move forward in the model
                loss+=sum(self.model.losses)

            # Compute and apply gradients
            grads = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            # Update the progress bar
            if pretty_progressbar:
                epoch_progress.set_description("Loss: {0:.3e} | Epoch".format(loss))

            # Log the relevant data for internal use
            if logging:
                history.on_epoch_end(epoch=i,logs={"loss":loss.numpy(), "std":std.numpy()})

            # Log the data in tensorboard
            tf.summary.scalar('loss', data=loss, step=i)
            tf.summary.scalar('std',data=std,step=i)

            if logging:
                return history


