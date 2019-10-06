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
        # Last layer is a SVM
        NN_layers.append(keras.layers.Dense(sizes[-1]))
        NN_layers.append(keras.layers.Reshape((2,self.transform_size)))
        self.NN = keras.Sequential(NN_layers)
    
    def call(self,input):
        xA = input[:,:self.pass_through_size]
        xB = input[:,self.pass_through_size:self.flow_size]
        shift_rescale = self.NN(xA)
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