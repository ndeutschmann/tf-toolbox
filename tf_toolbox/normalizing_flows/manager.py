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




