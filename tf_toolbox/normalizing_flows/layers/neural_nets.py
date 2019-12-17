"""Implementation of neural nets for the transverse leg of coupling cells"""
import tensorflow.keras as keras
from numpy import prod

class GenericDNN(keras.Sequential):
    """A generic dense neural network with a list of layer widths and a specific output shape"""
    def __init__(self, *, layer_widths=[], input_size, output_shape,
                 layer_activation="relu", final_activation="exponential", l2_reg=0., dropout_rate=0.):

        all_layer_widths = layer_widths + [prod(output_shape)]

        dense_layers = []

        if len(layer_widths) == 0:
            dense_layers.append(
                keras.layers.Dense(all_layer_widths[0],
                                   input_shape=(input_size,),
                                   activation=final_activation,
                                   kernel_regularizer=keras.regularizers.l2(l2_reg)
                                   )
            )

        else:
            dense_layers.append(
                keras.layers.Dense(all_layer_widths[0],
                                   input_shape=(input_size,),
                                   activation=layer_activation,
                                   kernel_regularizer=keras.regularizers.l2(l2_reg)
                                   )
            )

            for width in all_layer_widths[1:-1]:
                dense_layers.append(
                    keras.layers.Dense(width,
                                       activation=layer_activation,
                                       kernel_regularizer=keras.regularizers.l2(l2_reg))
                )
                if dropout_rate>0.:
                    dense_layers.append(
                        keras.layers.Dropout(dropout_rate)
                    )

            dense_layers.append(
                keras.layers.Dense(
                    all_layer_widths[-1],
                    activation=final_activation,
                    kernel_regularizer=keras.regularizers.l2(l2_reg)))

        dense_layers.append(keras.layers.Reshape(output_shape))
        
        super(GenericDNN, self).__init__(dense_layers)


class RectangularDNN(GenericDNN):
    def __init__(self, *, width, depth, input_size, output_shape,
                 layer_activation="relu", final_activation="exponential", l2_reg=0., dropout_rate=0.):
        layer_widths = [width]*depth
        super(RectangularDNN, self).__init__(layer_widths=layer_widths,
                                             input_size=input_size,
                                             output_shape=output_shape,
                                             layer_activation=layer_activation,
                                             final_activation=final_activation,
                                             l2_reg=l2_reg,
                                             dropout_rate=dropout_rate)

class RectangularResBlock(keras.Model):
    """One dense layer with width W then depth-1 dense layers with width W and a skip connection"""
    def __init__(self, *, width, depth, input_size, output_shape,
                 layer_activation="relu", final_activation="exponential", l2_reg=0., dropout_rate=0.):
        assert depth > 1, "A resnet needs multiple layers"
        
        super(RectangularResBlock, self).__init__()
        
        self.first_layer = keras.layers.Dense(width,
                                              layer_activation,
                                              input_shape=(input_size,),
                                              kernel_regularizer=keras.regularizers.l2(l2_reg))

        self.rect_dnn = RectangularDNN(width=width,
                                       depth=depth-1,
                                       input_size=width,
                                       output_shape=(width,),
                                       layer_activation=layer_activation,
                                       final_activation=layer_activation,
                                       l2_reg=l2_reg,
                                       dropout_rate=dropout_rate)

        self.final_layer = RectangularDNN(width=width,
                                          depth=0,
                                          input_size=width,
                                          output_shape=output_shape,
                                          layer_activation=layer_activation,
                                          final_activation=final_activation,
                                          l2_reg=l2_reg,
                                          dropout_rate=dropout_rate)

    def call(self, inputs):
        x = self.first_layer(inputs)
        x = self.rect_dnn(x) + x
        return self.final_layer(x)