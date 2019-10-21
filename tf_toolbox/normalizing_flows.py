import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tqdm.autonotebook import tqdm



class AffineCoupling(keras.layers.Layer):
    def __init__(self, flow_size, pass_through_size, NN_layers):
        super(AffineCoupling, self).__init__()
        self.pass_through_size = pass_through_size
        self.flow_size = flow_size
        self.transform_size = flow_size - pass_through_size
        sizes = NN_layers+[(2*self.transform_size)]
        NN_layers = [keras.layers.Dense(sizes[0],input_shape=(pass_through_size,),activation="relu")]
        for size in sizes[1:-1]:
            NN_layers.append(
                keras.layers.Dense(size,activation="relu")
            )
        # Last layer will be exponentiated so tame it with a sigmoid
        NN_layers.append(keras.layers.Dense(sizes[-1],activation='sigmoid'))
        NN_layers.append(keras.layers.Reshape((2,self.transform_size)))
        self.NN = keras.Sequential(NN_layers)
    
    def call(self, input):
        xA = input[:,:self.pass_through_size]
        xB = input[:,self.pass_through_size:self.flow_size]
        shift_rescale = self.NN(xA)
        shift_rescale[:,1]=tf.exp(shift_rescale[:,1])
        yB = tf.math.multiply(xB,shift_rescale[:,1])+shift_rescale[:,0]
        jacobian = input[:,self.flow_size]
        jacobian*= tf.reduce_prod(shift_rescale[:,1],axis=1)
        return tf.concat((xA,yB,tf.expand_dims(jacobian,1)),axis=1)


class AddJacobian(keras.layers.Layer):
    def __init__(self,jacobian_value = tf.constant(1.)):
        super(AddJacobian,self).__init__()
        self.jacobian_value = jacobian_value
    
    def call(self, input):
        return tf.concat((input, tf.broadcast_to(self.jacobian_value,(input.shape[0],1)) ),axis=1)


# WIP
class PieceWiseLinear(keras.layers.Layer):
    def __init__(self, flow_size, pass_through_size, n_bins=10, nn_layers=[],reg=0.):
        super(PieceWiseLinear,self).__init__()
        self.pass_through_size = pass_through_size
        self.flow_size = flow_size
        self.transform_size = flow_size - pass_through_size
        self.n_bins = n_bins
        sizes = nn_layers + [(n_bins * self.transform_size)]
        nn_layers = [keras.layers.Dense(sizes[0], input_shape=(pass_through_size,), activation="relu",kernel_regularizer=tf.keras.regularizers.l2(reg))]
        for size in sizes[1:-1]:
            nn_layers.append(keras.layers.Dense(size,activation="relu",kernel_regularizer=tf.keras.regularizers.l2(reg)))
            #nn_layers.append(keras.layers.Dropout(0.05))
            nn_layers.append(keras.layers.BatchNormalization())

        nn_layers.append(keras.layers.Dense(sizes[-1], activation="sigmoid",kernel_regularizer=tf.keras.regularizers.l2(reg)))
        nn_layers.append(keras.layers.Reshape((self.transform_size, n_bins)))
        self.NN = keras.Sequential(nn_layers)
        self.inverse = InversePieceWiseLinear(self)

    def call(self, x):
        xA = x[:, :self.pass_through_size]
        xB = x[:, self.pass_through_size:self.flow_size]
        jacobian = tf.expand_dims(x[:, self.flow_size], axis=-1)
        Q = self.NN(xA)
        Qsum = tf.cumsum(Q, axis=-1)
        Qnorms = tf.expand_dims(Qsum[:, :, -1], axis=-1)
        Q /= Qnorms/self.n_bins
        Qsum /= Qnorms
        Qsum = tf.pad(Qsum,tf.constant([[0, 0], [0, 0], [1, 0]]))
        alphas = xB*self.n_bins
        bins = tf.math.floor(alphas)
        alphas -= bins
        alphas/=self.n_bins
        bins = tf.cast(bins, tf.int32)
        # Sum of the integrals of the bins
        cdf_int_part = tf.gather(Qsum, tf.expand_dims(bins, axis=-1), batch_dims=-1, axis=-1)
        cdf_float_part = tf.gather(Q, tf.expand_dims(bins, axis=-1), batch_dims=-1, axis=-1)
        cdf = tf.reshape((cdf_float_part*tf.expand_dims(alphas, axis=-1))+cdf_int_part, cdf_int_part.shape[:-1])
        jacobian *= tf.reduce_prod(cdf_float_part, axis=-2)
        return tf.concat((xA,cdf, jacobian), axis=-1)

class InversePieceWiseLinear(keras.layers.Layer):
    def __init__(self, forward_layer):
        super(InversePieceWiseLinear, self).__init__()
        self.pass_through_size = forward_layer.pass_through_size
        self.flow_size = forward_layer.flow_size
        self.transform_size = forward_layer.flow_size - forward_layer.pass_through_size
        self.n_bins = forward_layer.n_bins
        self.NN = forward_layer.NN

    def call(self,y):
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
        offsets = tf.squeeze(tf.gather(paddedQsum, ybins, batch_dims=-1),axis=-1)
        slopes = tf.squeeze(tf.gather(Q, ybins, batch_dims=-1),axis=-1)
        xB = (yB - offsets) / slopes + (tf.squeeze(tf.cast(ybins, tf.float32),axis=-1)) / self.n_bins
        jacobian *= tf.expand_dims(tf.reduce_prod(1 / slopes, axis=-1), axis=-1)
        return tf.concat((yA, xB, jacobian), axis=-1)

class RollLayer(keras.layers.Layer):
    def __init__(self,shift):
        super(RollLayer,self).__init__()
        self.shift = shift
        self.inverse = InverseRollLayer(self)

    def call(self,x):
        return tf.concat((tf.roll(x[:,:-1],self.shift,axis=-1),x[:,-1:]),axis=-1)


class InverseRollLayer(keras.layers.Layer):
    def __init__(self,roll_layer):
        super(InverseRollLayer,self).__init__()
        self.shift = roll_layer.shift

    def call(self,x):
        return tf.concat((tf.roll(x[:,:-1],-self.shift,axis=-1),x[:,-1:]),axis=-1)


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
                loss = tf.reduce_mean(J*fX*tf.math.log(J+1e-3))
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
