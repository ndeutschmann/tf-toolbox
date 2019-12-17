import tensorflow as tf
import tensorflow.keras as keras


class AddJacobian(keras.layers.Layer):
    def __init__(self, jacobian_value=tf.constant(1.)):
        super(AddJacobian, self).__init__()
        self.jacobian_value = jacobian_value

    def call(self, input):
        return tf.concat((input, tf.broadcast_to(self.jacobian_value, (input.shape[0], 1))), axis=1)


class RollLayer(keras.layers.Layer):
    def __init__(self, shift):
        super(RollLayer, self).__init__()
        self.shift = shift
        self.inverse = InverseRollLayer(self)

    def call(self, x):
        return tf.concat((tf.roll(x[:, :-1], self.shift, axis=-1), x[:, -1:]), axis=-1)


class InverseRollLayer(keras.layers.Layer):
    def __init__(self, roll_layer):
        super(InverseRollLayer, self).__init__()
        self.shift = roll_layer.shift

    def call(self, x):
        return tf.concat((tf.roll(x[:, :-1], -self.shift, axis=-1), x[:, -1:]), axis=-1)


class CenterHyperCube(keras.layers.Layer):
    def __init__(self):
        super(CenterHyperCube, self).__init__()
    def call(self,x):
        return x-0.5