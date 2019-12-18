import tensorflow as tf
from tensorflow import keras
from .neural_nets import GenericDNN,RectangularDNN,RectangularResBlock

class AffineCoupling(keras.layers.Layer):
    def __init__(self, flow_size, pass_through_size, NN_layers):
        super(AffineCoupling, self).__init__()
        self.pass_through_size = pass_through_size
        self.flow_size = flow_size
        self.transform_size = flow_size - pass_through_size
        sizes = NN_layers + [(2 * self.transform_size)]
        NN_layers = [keras.layers.Dense(sizes[0], input_shape=(pass_through_size,), activation="relu")]
        for size in sizes[1:-1]:
            NN_layers.append(
                keras.layers.Dense(size, activation="relu")
            )
        # Last layer will be exponentiated so tame it with a sigmoid
        NN_layers.append(keras.layers.Dense(sizes[-1], activation='sigmoid'))
        NN_layers.append(keras.layers.Reshape((2, self.transform_size)))
        self.NN = keras.Sequential(NN_layers)

    def call(self, input):
        xA = input[:, :self.pass_through_size]
        xB = input[:, self.pass_through_size:self.flow_size]
        shift_rescale = self.NN(xA)
        shift_rescale[:, 1] = tf.exp(shift_rescale[:, 1])
        yB = tf.math.multiply(xB, shift_rescale[:, 1]) + shift_rescale[:, 0]
        jacobian = input[:, self.flow_size]
        jacobian *= tf.reduce_prod(shift_rescale[:, 1], axis=1)
        return tf.concat((xA, yB, tf.expand_dims(jacobian, 1)), axis=1)


class GeneralPieceWiseLinearCoupling(keras.layers.Layer):
    """Piecewise linear layer that takes an arbitrary neural network as its transverse
    leg."""
    def __init__(self,*, flow_size, pass_through_size, n_bins=10, nn):
        super(GeneralPieceWiseLinearCoupling, self).__init__()
        self.pass_through_size = pass_through_size
        self.flow_size = flow_size
        self.transform_size = flow_size - pass_through_size
        self.n_bins = n_bins

        # TODO: fix this check (fails due to shape not yet initialized although the model is supposed
        # TODO: to have an "input_shape" fixed)

        # assert nn.input_shape == (None, pass_through_size), "Transverse neural network input shape incorrect"
        # assert nn.output_shape == (None, self.transform_size, self.n_bins), "Transverse neural network output" \
        #                                                                            " shape incorrect"
        self.NN = nn

        self.inverse = InversePieceWiseLinearCoupling(self)

    def call(self, x):
        xA = x[:, :self.pass_through_size]
        xB = x[:, self.pass_through_size:self.flow_size]
        jacobian = tf.expand_dims(x[:, self.flow_size], axis=-1)
        Q = self.NN(xA)
        Qsum = tf.cumsum(Q, axis=-1)
        Qnorms = tf.expand_dims(Qsum[:, :, -1], axis=-1)
        Q /= Qnorms / self.n_bins
        Qsum /= Qnorms
        Qsum = tf.pad(Qsum, tf.constant([[0, 0], [0, 0], [1, 0]]))
        alphas = xB * self.n_bins
        bins = tf.math.floor(alphas)
        alphas -= bins
        alphas /= self.n_bins
        bins = tf.cast(bins, tf.int32)
        # Sum of the integrals of the bins
        cdf_int_part = tf.gather(Qsum, tf.expand_dims(bins, axis=-1), batch_dims=-1, axis=-1)
        cdf_float_part = tf.gather(Q, tf.expand_dims(bins, axis=-1), batch_dims=-1, axis=-1)
        cdf = tf.reshape((cdf_float_part * tf.expand_dims(alphas, axis=-1)) + cdf_int_part, cdf_int_part.shape[:-1])
        jacobian *= tf.reduce_prod(cdf_float_part, axis=-2)
        return tf.concat((xA, cdf, jacobian), axis=-1)


class InversePieceWiseLinearCoupling(keras.layers.Layer):
    def __init__(self, forward_layer):
        super(InversePieceWiseLinearCoupling, self).__init__()
        self.pass_through_size = forward_layer.pass_through_size
        self.flow_size = forward_layer.flow_size
        self.transform_size = forward_layer.flow_size - forward_layer.pass_through_size
        self.n_bins = forward_layer.n_bins
        self.NN = forward_layer.NN

    def call(self, y):
        yA = y[:, :self.pass_through_size]
        yB = y[:, self.pass_through_size:self.flow_size]
        jacobian = tf.expand_dims(y[:, self.flow_size], axis=-1)
        Q = self.NN(yA)
        Qsum = tf.cumsum(Q, axis=-1)
        Qnorms = tf.expand_dims(Qsum[:, :, -1], axis=-1)
        Q /= Qnorms / self.n_bins
        Qsum /= Qnorms
        ybins = tf.searchsorted(Qsum, tf.expand_dims(yB, axis=-1))
        paddedQsum = tf.pad(Qsum, [[0, 0], [0, 0], [1, 0]])
        offsets = tf.squeeze(tf.gather(paddedQsum, ybins, batch_dims=-1), axis=-1)
        slopes = tf.squeeze(tf.gather(Q, ybins, batch_dims=-1), axis=-1)
        xB = (yB - offsets) / slopes + (tf.squeeze(tf.cast(ybins, tf.float32), axis=-1)) / self.n_bins
        jacobian *= tf.expand_dims(tf.reduce_prod(1 / slopes, axis=-1), axis=-1)
        return tf.concat((yA, xB, jacobian), axis=-1)


class DNN_PieceWiseLinearCoupling(GeneralPieceWiseLinearCoupling):
    def __init__(self, flow_size, pass_through_size, n_bins=10, nn_layers=[], reg=0., dropout=0., layer_activation="relu",
                 final_activation="exponential"):
        nn = GenericDNN(layer_widths=nn_layers,
                        input_size=pass_through_size,
                        output_shape=(flow_size-pass_through_size,n_bins),
                        layer_activation=layer_activation,
                        final_activation=final_activation,
                        l2_reg=reg,dropout_rate=dropout)
        
        super(DNN_PieceWiseLinearCoupling, self).__init__(flow_size=flow_size,
                                                          pass_through_size=pass_through_size,
                                                          n_bins=n_bins,
                                                          nn=nn)


class RectDNN_PieceWiseLinearCoupling(GeneralPieceWiseLinearCoupling):
    def __init__(self,*, flow_size, pass_through_size, n_bins=10, width, depth, reg=0., dropout=0.,
                 layer_activation="relu", final_activation="exponential"):
        nn = RectangularDNN(depth=depth,width=width,
                        input_size=pass_through_size,
                        output_shape=(flow_size - pass_through_size, n_bins),
                        layer_activation=layer_activation,
                        final_activation=final_activation,
                        l2_reg=reg, dropout_rate=dropout)

        super(RectDNN_PieceWiseLinearCoupling, self).__init__(flow_size=flow_size,
                                                          pass_through_size=pass_through_size,
                                                          n_bins=n_bins,
                                                          nn=nn)

class RectResnet_PieceWiseLinearCoupling(GeneralPieceWiseLinearCoupling):
    def __init__(self,*, flow_size, pass_through_size, n_bins=10, width, depth, reg=0., dropout=0.,
                 layer_activation="relu", final_activation="exponential"):
        nn = RectangularResBlock(depth=depth,width=width,
                        input_size=pass_through_size,
                        output_shape=(flow_size - pass_through_size, n_bins),
                        layer_activation=layer_activation,
                        final_activation=final_activation,
                        l2_reg=reg, dropout_rate=dropout)

        super(RectResnet_PieceWiseLinearCoupling, self).__init__(flow_size=flow_size,
                                                          pass_through_size=pass_through_size,
                                                          n_bins=n_bins,
                                                          nn=nn)
