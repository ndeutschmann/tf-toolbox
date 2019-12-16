import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tqdm.autonotebook import tqdm
from .layers.coupling_cells import PieceWiseLinear
from .layers.misc import AddJacobian


class NormalizingFlow:

    format_input = AddJacobian()

    def __init__(self,flow_size,layers):
        self.flow_size = flow_size
        self.model = keras.Sequential()
        self.inverse_model = keras.Sequential()
        self.build_model(layers)
        self.build_inverse()

    def build_model(self,layers):
        for l in layers:
            if isinstance(l,PieceWiseLinear):
                assert self.flow_size == l.flow_size
            self.model.add(l)


    def build_inverse(self):
        self.inverse_model = keras.Sequential([l.inverse for l in reversed(self.model.layers)])

    def __call__(self,x):
        return self.model(x)

    def add(self,layer,rebuild_inverse=True):
        if isinstance(layer,PieceWiseLinear):
            assert self.flow_size == layer.flow_size
        self.model.add(layer)
        self.build_inverse()


    def train_variance(self,f ,n_batch = 10000,n_steps=1,n_epochs=10,*, optimizer,log_var=True):
        if log_var:
            var_log = []
        epoch_progress = tqdm(range(n_epochs),leave=False,desc="Loss: {0:.3e} | Epoch".format(0.))
        step_progress = tqdm_recycled(range(n_steps),desc="Step: ")
        for i in epoch_progress:
            # Generate some data
            Xs,fXs = self.generate_data_batches(f,n_batch=n_batch,n_steps=n_steps)
            for step in step_progress:
                X = tf.stop_gradient(Xs[step])
                fX = tf.stop_gradient(fXs[step])
                with tf.GradientTape() as tape:
                    J = self.inverse_model(self.format_input(X))[:, -1]
                    loss = tf.math.log(tfp.stats.variance(fX/J))
                    loss+=sum(self.inverse_model.losses)
                grads = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                if log_var:
                    var_log.append(loss)
                epoch_progress.set_description("Loss: {0:.3e} | Epoch".format(loss))
        step_progress.really_close()
        if log_var:
            return(var_log)

    def train_variance_forward(self,f ,n_batch = 10000,n_steps=1,n_epochs=10,*, optimizer,log_var=True):
        if log_var:
            var_log = []
        epoch_progress = tqdm(range(n_epochs),leave=False,desc="Loss: {0:.3e} | Epoch".format(0.))
        for i in epoch_progress:
            with tf.GradientTape() as tape:
                XJ = self.model(
                    self.format_input(
                        tf.random.uniform((n_batch, self.flow_size), 0, 1)
                    )
                )
                X = tf.stop_gradient(XJ[:,:-1])
                J = XJ[:,-1]
                fX = f(X)
                loss = tf.math.log(tfp.stats.variance(fX*J))
                loss+=sum(self.inverse_model.losses)
            grads = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            if log_var:
                var_log.append(loss)
            epoch_progress.set_description("Loss: {0:.3e} | Epoch".format(loss))
        if log_var:
            return(var_log)

    def train_KL_forward(self,f ,n_batch = 10000,n_steps=1,n_epochs=10,*, optimizer,log_var=True):
        if log_var:
            var_log = []
        epoch_progress = tqdm(range(n_epochs),leave=False,desc="Loss: {0:.3e} | Epoch".format(0.))
        for i in epoch_progress:
            with tf.GradientTape() as tape:
                XJ = self.model(
                    self.format_input(
                        tf.random.uniform((n_batch, self.flow_size), 0, 1)
                    )
                )
                X = tf.stop_gradient(XJ[:,:-1])
                J = XJ[:,-1]
                fX = f(X)
                loss = tf.reduce_mean(J*fX*tf.math.log(J))
                loss+=sum(self.inverse_model.losses)
            grads = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            if log_var:
                var_log.append(loss)
            epoch_progress.set_description("Loss: {0:.3e} | Epoch".format(loss))
        if log_var:
            return(var_log)

    def train_KL_flat(self,f ,n_batch = 10000,n_steps=1,n_epochs=10,*, optimizer,log_var=True):
        if log_var:
            var_log = []
        epoch_progress = tqdm(range(n_epochs),leave=False,desc="Loss: {0:.3e} | Epoch".format(0.))
        for i in epoch_progress:
            with tf.GradientTape() as tape:
                X=tf.random.uniform((n_batch, self.flow_size), 0, 1)
                J = self.inverse_model(self.format_input(X))[:,-1]
                fX = f(X)
                loss = tf.reduce_mean(-fX/J*tf.math.log(J))
                loss+=sum(self.inverse_model.losses)
            grads = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            if log_var:
                var_log.append(loss)
            epoch_progress.set_description("Loss: {0:.3e} | Epoch".format(loss))
        if log_var:
            return(var_log)

    def train_KL_backward(self,f ,n_batch = 10000,n_steps=1,n_epochs=10,*, optimizer,log_var=True):
        if log_var:
            var_log = []
        epoch_progress = tqdm(range(n_epochs),leave=False,desc="Loss: {0:.3e} | Epoch".format(0.))
        step_progress = tqdm_recycled(range(n_steps),desc="Step: ")
        for i in epoch_progress:
            # Generate some data
            Xs,fXs = self.generate_data_batches(f,n_batch=n_batch,n_steps=n_steps)
            for step in step_progress:
                X = tf.stop_gradient(Xs[step])
                fX = tf.stop_gradient(fXs[step])
                with tf.GradientTape() as tape:
                    J = self.inverse_model(self.format_input(X))[:, -1]
                    loss = tf.reduce_mean(-fX / J * tf.math.log(J))
                    loss += sum(self.inverse_model.losses)
                grads = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                if log_var:
                    var_log.append(loss)
                epoch_progress.set_description("Loss: {0:.3e} | Epoch".format(loss))
        step_progress.really_close()
        if log_var:
            return(var_log)

    def generate_data_batches(self,f,n_batch = 10000,n_steps=1):
        # Get a batch of random latent space points,
        # add a jacobian, generate a phase space sample using the model
        # then get rid of the jacobian
        X = self.model(
            self.format_input(
                tf.random.uniform((n_batch, self.flow_size), 0, 1)
            )
        )[:, :-1]
        # Evaluate the function over the function
        fX = f(X)
        Xs = tf.split(X,n_steps,axis=0)
        fXs = tf.split(fX,n_steps,axis=0)
        return (Xs,fXs)


class tqdm_recycled(tqdm):

    def close(self):
        self.reset()

    def really_close(self):
        try:
            self.sp(close=True)
        except AttributeError as e:
            pass
            #print("tqdm_recycled objects can be 'really closed' only when used in a jupyter notebook.")
            #raise e
