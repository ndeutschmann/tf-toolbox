import tensorflow as tf
import tensorflow.keras as keras
from tensorboard.plugins.hparams import api as hp
from tqdm.autonotebook import tqdm
from tf_toolbox.training.misc import tqdm_recycled
from .layers.coupling_cells import RectDNN_PieceWiseLinearCoupling,RectResnet_PieceWiseLinearCoupling
from .layers.misc import AddJacobian, RollLayer

from ..training.tf_managers.models import StandardModelManager


class GenericFlowManager(StandardModelManager):
    """Generic flow model manager implementing architecture-independent methods (training, etc)"""

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
                 batch_size_domain = (100,1000000),
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
            batch_size_domain ():
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

            "batch_size": hp.HParam("batch_size", domain=hp.IntInterval(*batch_size_domain),
                                      display_name="Batch size"),

            "use_batch_norm": hp.HParam("use_batch_norm", domain=hp.Discrete([True,False]),display_name="BatchNorm")
        }

        self._metrics = {
            "std": hp.Metric("std", display_name="Integrand standard deviation")
        }

        self.optimizer_object = None

    def train_model(self, train_mode = "variance_backward_staggered", batch_size = 10000, minibatch_size=10000, epochs=10, epoch_start=0,
                    logging=True, pretty_progressbar=True, save_best = True,
                    *, f, logdir, logger_functions, **train_opts):
        """Training method that dispatches the model into the different training modes.

        Training modes are implemented as methods named with the convention _train_{train_mode}

        Currently implemented modes are
            * variance_forward
                specific options are (TODO)
            * TODO

        Args:
            f (): function to train on
            logging (): return loss and accuracy histories?
            logdir (): where to log tb data
            train_mode ():
            batch_size ():
            epochs ():
            epoch_start():
            pretty_progressbar ():
            minibatch_size ():
            save_best():
            logger_functions():
            **train_opts ():


        Returns:
            keras.callbacks.History

        """
        try:
            trainer = getattr(self,"_train_"+train_mode)
        except AttributeError as error:
            raise AttributeError("The train mode %s does not exist."%train_mode)

        assert hasattr(self,"optimizer_object") and getattr(self,"optimizer_object") is not None, "This model manager " \
                                                                                                  "needs an " \
                                                                                                  "optimizer_object " \
                                                                                                  "to run "

        # if we save a checkpoint for the best model in training history, initialize the checkpoint and log the hparams
        if save_best:
            self.save_weights(logdir=logdir,prefix="best")

        return trainer(f, batch_size=batch_size, minibatch_size=minibatch_size, epochs=epochs, epoch_start=epoch_start,
                           logging=logging, pretty_progressbar=pretty_progressbar,
                           optimizer_object=self.optimizer_object, save_best=save_best, logdir=logdir,
                        logger_functions=logger_functions, **train_opts)

    def _train_variance_forward(self, f, batch_size = 10000, minibatch_size=10000, epochs=10, epoch_start=0,
                                logging=True, pretty_progressbar=True, save_best=True,
                                *, optimizer_object, logdir, logger_functions, **train_opts):
        """Train the model using the integrand variance as loss and compute the Jacobian in the forward pass
        (fixed latent space sample mapped to a phase space sample)
        See notes equation TODO

        Args:
            f ():
            batch_size ():
            minibatch_size():
            epochs ():
            epoch_start ():
            logging ():
            log_tb ():
            pretty_progressbar ():
            optimizer_object ():
            save_best ():
            logdir ():
            **train_opts ():

        Returns:

        """

        # Minibatch logic
        assert minibatch_size<=batch_size, "The minibatch size must be smaller than the batch size"
        n_minibatches = int(batch_size/minibatch_size)

        # Instantiate a pretty progress bar if needed
        if pretty_progressbar:
            epoch_progress = tqdm(range(epoch_start,epoch_start+epochs), leave=False, desc="Loss: {0:.3e} | Epoch".format(0.))
            # Instantiate a pretty progress bar for the minibatch loop if it is not trivial
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

        # Run the model once
        XJ = self.model(  # Pass through the model
            self.format_input(  # Append a unit Jacobian to each point
                tf.random.uniform((minibatch_size, self.n_flow), 0, 1)  # Generate a batch of points in latent space
            )
        )
        # Initialize tracking for checkpoints
        if save_best:
            fX = f(XJ[:,:-1])
            J = XJ[:,-1]
            best_std = tf.math.reduce_std(fX*J)

        self.model.build((minibatch_size, self.n_flow+1))
        variables = self.model.trainable_variables

        # Loop over epochs
        for i in epoch_progress:
            grads_cumul = [tf.zeros_like(variable) for variable in variables]
            loss_cumul = 0
            std_cumul = 0
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
                    X = tf.stop_gradient(XJ[:, :-1])
                    # # Apply function values and combine with the Jacobian (last entry of each X)
                    fXJ = tf.multiply(f(X), XJ[:, -1])

                    # The Monte Carlo integrand is fXJ: we minimize its variance up to the constant term
                    loss = tf.reduce_mean(tf.square(fXJ))
                    # Regularization losses are collected as we move forward in the model
                    loss += sum(self.model.losses)

                # Compute and apply gradients
                grads = tape.gradient(loss, self.model.trainable_variables)
                grads_cumul = [grads[k]+grads_cumul[k] for k in range(len(grads))]
                loss_cumul += loss
                std_cumul += tf.math.reduce_std(fXJ)
            grads_cumul = [g / n_minibatches for g in grads_cumul]
            loss_cumul /= n_minibatches
            std_cumul /= n_minibatches
            optimizer_object.apply_gradients(zip(grads_cumul, self.model.trainable_variables))

            # Update the progress bar
            if pretty_progressbar:
                epoch_progress.set_description("Loss: {0:.3e} | Epoch".format(loss_cumul))

            # Log the relevant data for internal use
            if logging:
                history.on_epoch_end(epoch=i,logs={"loss":float(loss_cumul), "std":float(std_cumul)})

            # Log the data
            # Logger function must take arguments as name,value,step
            for lf in logger_functions:
                lf('loss', float(loss_cumul), i)
                lf('std',  float(std_cumul),  i)

            if save_best and std_cumul < best_std:
                best_std = std_cumul
                self.save_weights(logdir=logdir,prefix="best")

        if isinstance(minibatch_progress,tqdm_recycled):
            minibatch_progress.really_close()
        if logging:
            return history

    def _train_variance_backward_staggered(self, f, batch_size = 10000, minibatch_size=10000, epochs=10, epoch_start=0,
                                           n_epochs_before_refresh=50,logging=True, pretty_progressbar=True,
                                           save_best=True, early_stopping=True,
                                           *, optimizer_object, logdir, logger_functions, **train_opts):
        """Train the model using the integrand variance as loss and compute the Jacobian in the forward pass
        (fixed latent space sample mapped to a phase space sample)
        See notes equation TODO

        Args:
            f ():
            batch_size ():
            minibatch_size():
            epochs ():
            epoch_start ():
            logging ():
            log_tb ():
            pretty_progressbar ():
            optimizer_object ():
            save_best ():
            logdir ():
            **train_opts ():

        Returns:

        """

        # Minibatch logic
        assert minibatch_size<=batch_size, "The minibatch size must be smaller than the batch size"
        n_minibatches = int(batch_size/minibatch_size)

        # Instantiate a pretty progress bar if needed
        if pretty_progressbar:
            epoch_progress = tqdm(range(epoch_start,epoch_start+epochs), leave=False, desc="Loss: {0:.3e} | Epoch".format(0.))
            # Instantiate a pretty progress bar for the minibatch loop if it is not trivial
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

        # Run the model once
        XJ = self.model(  # Pass through the model
            self.format_input(  # Append a unit Jacobian to each point
                tf.random.uniform((minibatch_size, self.n_flow), 0, 1)  # Generate a batch of points in latent space
            )
        )
        # Initialize tracking for checkpoints
        if save_best:
            fX = f(XJ[:,:-1])
            J = XJ[:,-1]
            best_std = tf.math.reduce_std(fX*J)

        self.model.build((minibatch_size, self.n_flow+1))
        variables = self.model.trainable_variables

        # TODO: Set this parameter as an "official" hyperparameter
        # Short term plan: refactor these managers to have one class for each training method
        # This can then be a hyperparameter of the staggered class
        # Long term plan: separate the training from the model manager and attach it at creation like the optimizer
        # -> a HP of the model manager is then the training mode
        # n_renew will be a HP of the separate training mode / training manager
        n_renew = n_epochs_before_refresh

        # Early stopping: finish training if five subsequent steps see the loss increasing
        if early_stopping:
            n_steps_with_increase = 0
        # Now we just initialize the loss
        loss_cumul = 0

        # Loop over epochs
        for i in epoch_progress:
            grads_cumul = [tf.zeros_like(variable) for variable in variables]

            if early_stopping:
                old_loss = loss_cumul

            loss_cumul = 0
            std_cumul = 0
            # Every n_renew, refresh our data batch
            if i%n_renew == 0:
                print("regenerating a new sample")
                XJ = tf.stop_gradient(self.model(  # Pass through the model
                    self.format_input(  # Append a unit Jacobian to each point
                        tf.random.uniform((batch_size, self.n_flow), 0, 1)
                        # Generate a batch of points in latent space
                    )
                ))
                # Apply function values and multiply by jacobian
                fX = f(XJ[:, :-1])
                fXs = tf.split(fX,n_minibatches)
                Xs = tf.stop_gradient(tf.split(XJ[:, :-1], n_minibatches))
                Js = tf.stop_gradient(tf.split(XJ[:, -1], n_minibatches))
                fX2Js = tf.split(XJ[:, -1]*fX**2,n_minibatches)

            for j in minibatch_progress:
                with tf.GradientTape() as tape:
                    Jinv = self._inverse_model(self.format_input(Xs[j]))[:,-1]
                    loss=tf.math.reduce_mean((fX2Js[j]/Jinv))

                # Compute and apply gradients
                grads = tape.gradient(loss, self.model.trainable_variables)
                grads_cumul = [grads[k]+grads_cumul[k] for k in range(len(grads))]
                loss_cumul += loss
                std = tf.sqrt(loss-tf.math.reduce_mean((fXs[j]*Js[j]))**2)
                std_cumul += std
            grads_cumul = [g / minibatch_size for g in grads_cumul]
            loss_cumul /= n_minibatches
            std_cumul /= n_minibatches
            optimizer_object.apply_gradients(zip(grads_cumul, self.model.trainable_variables))

            # Update the progress bar
            if pretty_progressbar:
                epoch_progress.set_description("Loss: {0:.3e} | Epoch".format(loss_cumul))

            # Log the relevant data for internal use
            if logging:
                history.on_epoch_end(epoch=i,logs={"loss":float(loss_cumul.numpy()), "std":float(std_cumul.numpy())})

            # Log the data
            # Logger function must take arguments as name,value,step
            for lf in logger_functions:
                lf('loss', float(loss_cumul), i)
                lf('std',  float(std_cumul),  i)

            if std_cumul < best_std and save_best:
                self.save_weights(logdir=logdir, prefix="best")
                best_std = std_cumul

            # Early stopping logic
            if early_stopping:
                # Check if the loss is going down
                if loss_cumul > old_loss:
                    n_steps_with_increase+=1
                else:
                    n_steps_with_increase=0
                # If the loss has been going up, stop training
                if n_steps_with_increase>5:
                    break

        if isinstance(minibatch_progress,tqdm_recycled):
            minibatch_progress.really_close()
        if logging:
            return history


class PWLinearRectRollingManager(GenericFlowManager):
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
                     use_batch_norm=True, # Not set as an 'official' hyperparam TODO
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
            self._model.add(
                RectDNN_PieceWiseLinearCoupling(flow_size=self.n_flow,
                                                pass_through_size=n_pass_through,
                                                n_bins=n_bins, width=nn_width,depth=nn_depth,
                                                reg=l2_reg, dropout=dropout_rate, layer_activation=nn_activation,
                                                use_batch_norm=use_batch_norm)
            )
            self._model.add(RollLayer(roll_step))

        self._inverse_model = keras.Sequential([l.inverse for l in reversed(self._model.layers)])

        self.optimizer_object = optimizer_object

        # Do one pass forward:
        self._model(
            self.format_input(
                tf.random.uniform((1,self.n_flow),0.,1.)
            )
        )

class PWLinearRectResnetRollingManager(PWLinearRectRollingManager):
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
                     use_batch_norm=True,
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
            self._model.add(
                RectResnet_PieceWiseLinearCoupling(flow_size=self.n_flow,
                                                   pass_through_size=n_pass_through,
                                                   n_bins=n_bins, width=nn_width,depth=nn_depth,
                                                   reg=l2_reg, dropout=dropout_rate, layer_activation=nn_activation,
                                                   use_batch_norm=use_batch_norm)
            )
            self._model.add(RollLayer(roll_step))

        self._inverse_model = keras.Sequential([l.inverse for l in reversed(self._model.layers)])

        self.optimizer_object = optimizer_object

        # Do one pass forward:
        self._model(
            self.format_input(
                tf.random.uniform((1,self.n_flow),0.,1.)
            )
        )
