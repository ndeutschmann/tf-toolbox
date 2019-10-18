#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow.keras as keras
from tf_toolbox.normalizing_flows import *


# In[2]:


@tf.function
def circle(x):
    return tf.exp( - (tf.norm(x-0.5,axis=-1)-0.3)**2/0.2**2 )


# In[3]:


n_flow = 2
n_pass = 1
n_bins = 10
layer_size = 300
n_layers=10

NF = NormalizingFlow(2,
    [
        PieceWiseLinear(n_flow,n_pass,n_bins,[layer_size]*n_layers)
    ])


# In[ ]:


optim = keras.optimizers.Adam(lr=1e-4)
variances = NF.train_variance(circle,n_batch=10,optimizer=optim)

