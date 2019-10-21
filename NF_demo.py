#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow.keras as keras
from tf_toolbox.normalizing_flows import *
import matplotlib.pyplot as plt


# In[2]:


@tf.function
def circle(x):
    return tf.exp( - (tf.norm(x-0.5,axis=-1)-0.3)**2/0.2**2 )


# In[3]:


n_flow = 2
n_pass = 1
n_bins = 10
layer_size = 20
n_layers=10

NF = NormalizingFlow(2,
    [
        #---- Block
        PieceWiseLinear(n_flow,n_pass,n_bins,[layer_size]*n_layers), #PW-linear coupling on variable 1
        RollLayer(1), # Circular permutation of the variables (i.e. swap x1,x2)
        PieceWiseLinear(n_flow,n_pass,n_bins,[layer_size]*n_layers), #PW-linear coupling on local variable 1 (i.e. x2)
        RollLayer(1), # Circular permutation of the variables (i.e. swap x1,x2)
        #---- End Block
    ])


# In[4]:


optim = keras.optimizers.Adam(lr=1e-4)
variances = NF.train_variance(circle,n_epochs=70,n_steps=10,n_batch=100000,optimizer=optim)

# In[5]:
fig = plt.figure(figsize=(4, 8))
fig.add_subplot(211)
plt.plot(variances)
X,fX=NF.generate_data_batches(circle)
ax=fig.add_subplot(212)
plt.hist2d(X[0][:,0],X[0][:,1])
ax.set_aspect(aspect=1.)
ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    left=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False)

