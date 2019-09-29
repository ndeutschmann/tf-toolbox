from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GaussianNoise, BatchNormalization, Activation
import numpy as np
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard


class DenseLayeredAutoencoder:
    """An autoencoder based on a sequence of dense layers
    This model uses gaussian noise normalization on the input and batch normalization on the latent layer.
    The model is stored in self.model, and the encoder and decoder in self.encoder and self.decoder
    Keras model methods fit,evaluate,predict and summary are exposed as methods and will pass
    any inputs to the model's corresponding methods.

    This model was designed to play with the MNIST dataset to try and use the latent space as a generator,
    with the expectation that batch normalization would allow gaussian noise input to generate convincing
    numbers.
    To this end, self.generate_image() will generate an image from random noise.
    """
    def __init__(self, data_size=(28 * 28), latent_size=10, encoding_sequence=[], decoding_sequence=None,
                 activation="relu", tensorboard_logging=True, optimizer=None):
        self.latent_size = latent_size
        self.model = Sequential()
        self.encoder = Sequential()
        self.decoder = Sequential()

        # Encoding sequence
        self.model.add(GaussianNoise(0.1, input_shape=(data_size,)))
        for size in encoding_sequence:
            self.model.add(Dense(size, activation=activation))
            self.encoder.add(self.model.layers[-1])

        # Reach the latent space
        self.model.add(Dense(latent_size))
        self.encoder.add(self.model.layers[-1])
        self.model.add(BatchNormalization())
        self.encoder.add(self.model.layers[-1])
        self.model.add(Activation('sigmoid'))
        self.decoder.add(self.model.layers[-1])

        # Decoding sequence
        if decoding_sequence is None:
            _decoding_sequence = reversed(encoding_sequence)
        for size in _decoding_sequence:
            self.model.add(Dense(size, activation=activation))
            self.decoder.add(self.model.layers[-1])
        # Final linear decoder
        self.model.add(Dense(data_size, activation="sigmoid"))
        self.decoder.add(self.model.layers[-1])

        # Logging
        self.logging = False
        if tensorboard_logging:
            self.logging = True
            logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            self.tensorboard_callback = TensorBoard(log_dir=logdir)

        # Set the optimizer
        if optimizer is None:
            _optimizer = "adam"
        else:
            _optimizer = optimizer

        # Compile the model
        self.model.compile(loss='mse',
                           optimizer=_optimizer)

        # Compile encoder and decoders, loss and optimizers are irrelevant as these models are not trained.
        self.encoder.compile(loss='mse',
                             optimizer="adam")
        self.decoder.compile(loss='mse',
                             optimizer="adam")

    def fit(self, X_train, batch_size=500, nb_epoch=50, verbose=1, **kwargs):
        if self.logging:
            self.model.fit(X_train, X_train,
                           batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose, shuffle=True,
                           callbacks=[self.tensorboard_callback], **kwargs)
        else:
            self.model.fit(X_train, X_train,
                           batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True, verbose=verbose, **kwargs)

    def evaluate(self, X_test, verbose=0, **kwargs):
        return self.model.evaluate(X_test, verbose=verbose, **kwargs)

    def predict(self,*args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def summary(self):
        self.model.summary()

    def generate_image(self, latent_vector=None, mean=0., std=1., n_images=1):
        if latent_vector is None:
            latent_data = np.random.normal(0., 1., (n_images, self.latent_size))
        else:
            latent_data = latent_vector
        return self.decoder.predict(latent_data)