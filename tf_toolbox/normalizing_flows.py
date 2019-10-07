class AffineCoupling(keras.layers.Layer):
    def __init__(self,flow_size,pass_through_size,NN_layers):
        super(NF_AffineCoupling,self).__init__()
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
    
    def call(self,input):
        xA = input[:,:self.pass_through_size]
        xB = input[:,self.pass_through_size:self.flow_size]
        shift_rescale = self.NN(xA)
        shift_rescale[:,1]=tf.exp(shift_rescale[:,1])
        yB = tf.math.multiply(xB,shift_rescale[:,1])+shift_rescale[:,0]
        jacobian = input[:,self.flow_size]
        jacobian*= tf.reduce_prod(shift_rescale[:,1],axis=1)
        return tf.concat((xA,yB,tf.expand_dims(jacobian,1)),axis=1)
        
class add_jacobian(keras.layers.Layer)           :
    def __init__(self,jacobian_value = tf.constant(1.)):
        super(add_jacobian,self).__init__()
        self.jacobian_value = jacobian_value
    
    def call(self,input):
        return tf.concat((input, tf.broadcast_to(self.jacobian_value,(input.shape[0],1)) ),axis=1)

# WIP
class PieceWiseLinear(keras.layers.Layer):
    def __init__(self,n_bins,flow_size,pass_through_size,NN_layers):
        super(NF_AffineCoupling,self).__init__()
        self.pass_through_size = pass_through_size
        self.flow_size = flow_size
        self.transform_size = flow_size - pass_through_size
        self.n_bins
        sizes = NN_layers+[(n_bins*self.transform_size)]
        NN_layers = [keras.layers.Dense(sizes[0],input_shape=(pass_through_size,),activation="relu")]
        for size in sizes[1:-1]:
            NN_layers.append(
                keras.layers.Dense(size,activation="relu")
            )

        NN_layers.append(keras.layers.Dense(sizes[-1],activation="sigmoid"))
        NN_layers.append(keras.layers.Reshape((self.transform_size,n_bins)))
        self.NN = keras.Sequential(NN_layers)
    
    def call(self,input):
        xA = input[:,:self.pass_through_size]
        xB = input[:,self.pass_through_size:self.flow_size]
        jacobian = tf.expand_dims(input[:,self.flow_size],axis=-1)
        Q = NN(xA)
        Qsum=tf.cumsum(Q,axis=-1)
        Qnorms = tf.expand_dims(Qsum[:,:,-1],axis=-1)
        Q/=Qnorms
        Qsum/=Qnorms
        Qsum=tf.pad(Qsum,tf.constant([[0,0],[0,0],[1,0]]))
        alphas = xB*n_bins
        bins = tf.math.floor(alphas)
        alphas -= bins
        bins = tf.cast(bins,tf.int32)
        cdf_int_part =tf.gather(Qsum,tf.expand_dims(bins,axis=-1),batch_dims=-1,axis=-1)
        cdf_float_part = tf.gather(Q,tf.expand_dims(bins,axis=-1),batch_dims=-1,axis=-1)
        cdf = tf.reshape((cdf_float_part*tf.expand_dims(alphas,axis=-1))+cdf_int_part,cdf_int_part.shape[:-1])
        jacobian *= tf.reduce_prod(cdf_float_part,axis=-2)
        return tf.concat((cdf,jacobian),axis=-1)
