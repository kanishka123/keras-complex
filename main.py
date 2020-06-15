import complexnn

import keras
from keras import models
from keras import layers
from keras import optimizers
import numpy as np 
import matplotlib.pyplot as plt 

# import keras.backend as KB
# import keras.engine as KE
# import keras.layers as KL
# import keras.optimizers as KO
# import theano as T
# import theano.ifelse as TI
# import theano.tensor as TT
# import theano.tensor.fft as TTF


L = 19
r = np.random.normal(0.8, size=(L,))
i = np.random.normal(0.8, size=(L,))
x = r+i*1j
R = np.fft.rfft(r, norm="ortho")
I = np.fft.rfft(i, norm="ortho")
X = np.fft.fft(x, norm="ortho")

input_shape = [128,32,32]

model = models.Sequential()

model.add(complexnn.conv.ComplexConv2D(32, (3, 3), activation='modrelu', padding='same', input_shape=input_shape))
model.add(complexnn.bn.ComplexBatchNormalization())
model.add(layers.MaxPooling2D((2, 2), padding='same'))

model.compile(optimizer=optimizers.Adam(), loss='mse')
