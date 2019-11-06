import tensorflow as tf
import tensorflow.keras as keras
import tensorboard.plugins.hparams.api as hp
import tensorflow_probability as tfp
from tqdm.autonotebook import tqdm
from tf_toolbox.training.misc import tqdm_recycled
from .layers import AddJacobian, PieceWiseLinear, RollLayer
import tf_toolbox.training.abstract_managers as AM

class RollingPWlinearNormalizingFlowManager(AM.ModelManager):
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
                 *,
                 n_flow: int,
                 n_pass_through_domain = None,
                 n_cells_domain=(1,10),
                 n_bins_domain = (2,10),
                 nn_width_domain = None,
                 nn_depth_domain = (1,10),
                 nn_activation_domain = ("relu",),
                 roll_step_domain = None,
                 l2_reg_domain = (0.,1.),
                 dropout_rate_domain = (0.,1.),
                 n_batch_domain = (100,1000000),
                 **init_opts
):
        """

        Args:
            n_flow ():
            n_pass_through_domain ():
            n_cells_domain ():
            n_bins_domain ():
            nn_width_domain ():
            nn_depth_domain ():
            nn_activation_domain ():
            roll_step_domain ():
            l2_reg_domain ():
            dropout_rate_domain ():
            n_batch_domain ():
            **init_opts ():
        """
        self.n_flow = n_flow
        self._model = None
        self._inverse_model = None

        # Some domains do not have an explicit default value as we want them to be dependent on other domains
        if n_pass_through_domain is None:
            _n_pass_through_domain = [1,n_flow-1]
        else: _n_pass_through_domain=n_pass_through_domain

        if nn_width_domain is None:
            _nn_width_domain = [n_bins_domain[0],5*n_bins_domain[1]]
        else: _nn_width_domain=nn_width_domain

        if roll_step_domain is None:
            _roll_step_domain = [1,n_flow-1]
        else: _roll_step_domain=roll_step_domain

        # **Hyperparameters**
        # Note that we include the number of batch points for training.
        # This is because we only have an estimator for the loss (which is defined as an integral)
        # And batch statistics has an impact on convergence
        # Pedagogical note: a contrario if we divide this sample into N minibatches and accumulate the gradients
        # before taking an optimizer step (typically for memory reasons)
        # TODO implement minibatches
        self._hparam = {
            "n_pass_through": hp.HParam("n_pass_through", domain=hp.IntInterval(*_n_pass_through_domain),display_name="# Pass"),

            "n_cells": hp.HParam("n_cells",domain=hp.IntInterval(*n_cells_domain),display_name="# Cells"),

            "n_bins": hp.HParam("n_bins", domain=hp.IntInterval(*n_bins_domain),display_name="# Bins"),

            "nn_width": hp.HParam("nn_width", domain=hp.IntInterval(*_nn_width_domain),display_name="NN width"),

            "nn_depth": hp.HParam("nn_depth", domain=hp.IntInterval(*nn_depth_domain),display_name="NN depth"),

            "nn_activation": hp.HParam("nn_activation", domain=hp.Discrete(nn_activation_domain),display_name="NN activ. fct."),

            "roll_step": hp.HParam("roll_step", domain=hp.IntInterval(*_roll_step_domain),display_name="Roll step"),

            "l2_reg": hp.HParam("l2_reg", domain=hp.RealInterval(*l2_reg_domain),display_name="L2 reg."),

            "dropout_rate": hp.HParam("dropout_rate", domain=hp.RealInterval(*dropout_rate_domain),display_name="Dropout rate"),

            "n_batch": hp.HParam("n_batch", domain=hp.IntInterval(*n_batch_domain),
                                      display_name="# Batch points"),

        }

        self._metrics = {
            "std": hp.Metric("std", display_name="Integrand standard deviation")
        }

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
                     optimizer_object,
                     **opts
                     ):
        """

        Args:
            n_pass_through ():
            n_cells ():
            n_bins ():
            nn_width ():
            nn_depth ():
            nn_activation ():
            roll_step ():
            l2_reg ():
            dropout_rate ():
            optimizer_object():
            **opts ():

        Returns:

        """

        self._model = keras.Sequential()

        for i_cell in range(n_cells):
            nn_layers = [nn_width]*nn_depth
            self._model.add(
                PieceWiseLinear(self.n_flow, n_pass_through, n_bins=n_bins, nn_layers=nn_layers,
                                reg=l2_reg, dropout=dropout_rate)
            )
            self._model.add(RollLayer(roll_step))

        self._inverse_model = keras.Sequential([l.inverse for l in reversed(self._model.layers)])

        self.optimizer_object = optimizer_object

    def train_model(self, train_mode = "variance_forward", n_batch = 10000, epochs=10, epoch_start=0, logging=True, log_tb=True,
                    pretty_progressbar=True, n_minibatches=1, *, f, logdir, hparam, **train_opts):
        """Training method that dispatches the model into the different training modes.

        Training modes are implemented as methods named with the convention _train_{train_mode}

        Currently implemented modes are
            * variance_forward
                specific options are (TODO)
            * TODO

        Note:
            * signature
            train_model(train_mode = "variance_forward", n_batch = 10000, n_epochs=10, logging=True, log_tb=True,
                    pretty_progressbar=True, *, f, optimizer, logdir, hparam)

        Args:
            f (): function to train on
            logging (): return loss and accuracy histories?
            log_tb (): log metrics and hparams into tensorboard (tb)?
            logdir (): where to log tb data
            hparam (): tb.plugins.hparam.Hparam-keyed dict for hparam logging in tb. TODO add YAML logging w/o tb
            train_mode ():
            n_batch ():
            epochs ():
            epoch_start():
            pretty_progressbar ():
            optimizer ():
            **train_opts ():

        Returns:
            keras.callbacks.History

        """
        try:
            trainer = getattr(self,"_train_"+train_mode)
        except AttributeError as error:
            raise AttributeError("The train mode %s does not exist."%train_mode)

        # if we use tensorboard, log the Hyperparameter values and start training
        if log_tb:
            with tf.summary.create_file_writer(logdir).as_default():
                hp.hparams(hparam)
                return trainer(f, n_batch=n_batch, epochs=epochs, epoch_start=epoch_start, logging=logging, log_tb=log_tb,
                               pretty_progressbar=pretty_progressbar, n_minibatches=n_minibatches, optimizer_object=self.optimizer_object, **train_opts)
        # Otherwise just start training
        else:
            return trainer(f, n_batch=n_batch, epochs=epochs, epoch_start=epoch_start, logging=logging, log_tb=log_tb,
                           pretty_progressbar=pretty_progressbar, n_minibatches=n_minibatches, optimizer_object=self.optimizer_object, **train_opts)

    def _train_variance_forward(self, f, n_batch = 10000, epochs=10, epoch_start=0, logging=True, log_tb=True,
                                pretty_progressbar=True, n_minibatches=1, *, optimizer_object, **train_opts):
        """Train the model using the integrand variance as loss and compute the Jacobian in the forward pass
        (fixed latent space sample mapped to a phase space sample)
        See notes equation TODO

        Args:
            f ():
            n_batch ():
            epochs ():
            logging ():
            log_tb ():
            pretty_progressbar ():
            optimizer ():
            **train_opts ():

        Returns:

        """

        # Instantiate a pretty launchbar if needed
        if pretty_progressbar:
            epoch_progress = tqdm(range(epoch_start,epoch_start+epochs), leave=False, desc="Loss: {0:.3e} | Epoch".format(0.))
            if n_minibatches>1:
                minibatch_progress = tqdm_recycled(range(n_minibatches), leave=False, desc="Step")
            else:
                minibatch_progress = range(n_minibatches)

        else:
            epoch_progress = range(epoch_start, epoch_start+epochs)
            minibatch_progress = range(n_minibatches)

        # Keep track of metric history if needed
        if logging:
            history = keras.callbacks.History()
            history.on_train_begin()

        assert n_minibatches>0,"n_minibatches must be strictly positive"
        minibatch_size = int(n_batch/n_minibatches)

        # Run the model once
        self.model(  # Pass through the model
            self.format_input(  # Append a unit Jacobian to each point
                tf.random.uniform((minibatch_size, self.n_flow), 0, 1)  # Generate a batch of points in latent space
            )
        )
        self.model.build((minibatch_size, self.n_flow+1))
        variables = self.model.trainable_variables

        # Loop over epochs
        for i in epoch_progress:
            grads_cumul = [tf.zeros_like(variable) for variable in variables]
            for j in minibatch_progress:
                with tf.GradientTape() as tape:
                    # Output a sample of (phase-space point, forward Jacobian)
                    XJ = self.model(                                            # Pass through the model
                        self.format_input(                                      # Append a unit Jacobian to each point
                            tf.random.uniform((minibatch_size, self.n_flow), 0, 1)  # Generate a batch of points in latent space
                        )
                    )
                    # Separate the points and their Jacobians:
                    # This sample is fixed, we optimize the Jacobian
                    X = tf.stop_gradient(XJ[:,:-1])
                    # The Jacobian is the last entry of each point
                    J = XJ[:,-1]
                    # Apply function values
                    fX = f(X)

                    # The Monte Carlo integrand is fX*J: we minimize its variance
                    loss = tf.math.reduce_variance(fX*J)
                    std = tf.math.sqrt(tf.stop_gradient(loss))
                    loss = tf.math.log(loss)
                    # Regularization losses are collected as we move forward in the model
                    loss += sum(self.model.losses)

                # Compute and apply gradients
                grads = tape.gradient(loss, self.model.trainable_variables)
                grads_cumul = [grads[i]+grads_cumul[i] for i in range(len(grads))]
            grads_cumul = [g/n_minibatches for g in grads_cumul]
            optimizer_object.apply_gradients(zip(grads_cumul, self.model.trainable_variables))

            # Update the progress bar
            if pretty_progressbar:
                epoch_progress.set_description("Loss: {0:.3e} | Epoch".format(loss))

            # Log the relevant data for internal use
            if logging:
                history.on_epoch_end(epoch=i,logs={"loss":loss.numpy(), "std":std.numpy()})

            # Log the data in tensorboard
            if log_tb:
                tf.summary.scalar('loss', data=loss, step=i)
                tf.summary.scalar('std',  data=std,  step=i)
        if isinstance(minibatch_progress,tqdm_recycled):
            minibatch_progress.really_close()
        if logging:
            return history


