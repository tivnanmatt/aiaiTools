
import sys
sys.path.append("../..")


import tensorflow as tf
import numpy as np
import aiaiTools
from aiaiTools.utils import Ones, Zeros, Rand, Randn, Range
import time

from skimage.transform import radon, iradon, resize
from skimage.data import shepp_logan_phantom

from matplotlib import pyplot as plt

nSample = 100
nVoxelX = 128
nVoxelY = 128
nPixel = 182
nView = 360
x_shape = [nVoxelX, nVoxelY]
y_shape = [nPixel, nView]
x_train = np.zeros([nSample,nVoxelX,nVoxelY],dtype=float)
y_train = np.zeros([nSample,nPixel,nView],dtype=float)

theta = np.linspace(0.0,180.0,nView,endpoint=False,dtype=float)
def FP(x):
    return radon(x, theta=theta, circle=False)
def BP(x):
    return iradon(x, theta=theta, circle=False, filter=None, output_size=nVoxelX)

for iSample in range(nSample):
    print('synthesizing training data ', iSample, ' / ', nSample)
    for iCircle in range(5):
        row = 63.5 + 10.0*np.random.randn()
        col = 63.5 + 10.0*np.random.randn()
        rad = 20.0 +  3.0*np.random.randn() 
        contrast = 10.0 +  2.0*np.random.randn() 
        x_train[iSample] = x_train[iSample] + contrast*aiaiTools.utils.RoiCircle2D(x_shape,row,col,rad)
    y_train[iSample] = FP(x_train[iSample])

x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

y_train = y_train + 50.0*Randn(y_train.shape)

myPermute1 = aiaiTools.layers.shape.Permute([1,0])
myFilter1D = aiaiTools.layers.linear.FourierOperator1D(aiaiTools.utils.Impulse1D([nPixel]),fft_length=[nPixel])
myRepeatedFilter1D = aiaiTools.layers.meta.Batch(myFilter1D, output_shape=[nPixel])
myPermute2 = aiaiTools.layers.shape.Permute([1,0])
myFilter = aiaiTools.layers.meta.Sequence(myPermute1, myRepeatedFilter1D, myPermute2)

myFilter = aiaiTools.layers.linear.FourierOperator2D(aiaiTools.utils.Impulse2D([nPixel,1]),fft_length=[nPixel,nView],frequencyDomain=False)

myBP = aiaiTools.layers.linear.NumpyLinearOperator(BP,FP,y_shape,x_shape,dynamic=True)

inputs = tf.keras.Input(shape=[nPixel,nView])
filt = aiaiTools.layers.meta.Batch(myFilter, output_shape=y_shape)(inputs)
fbp = aiaiTools.layers.meta.Batch(myBP, output_shape=x_shape,dynamic=True)(filt)
model = tf.keras.Model(inputs=inputs,outputs=fbp)

loss = 'mse'
learning_rate = 0.01
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer,loss=loss)
model.fit(y_train,x_train,batch_size=1,epochs=20)
model.fit(y_train,x_train,batch_size=nSample,epochs=5)
# model.fit(y_train,x_train,batch_size=nSample,epochs=50)
tmp = model.get_weights()
print(tmp.shape)

if True:
    x_predict = model.predict(y_train,batch_size=2)
    
    fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    fig.set_size_inches(18.5, 10.5)
    for iSample in range(nSample):
        print('plotting results ', iSample, ' / ', nSample)
        ax1.imshow(y_train[iSample],aspect='auto')
        img2 = ax2.imshow(x_train[iSample],aspect='equal')
        img2.set_clim([0.0, 30.0])
        img3 = ax3.imshow(x_predict[iSample],aspect='equal')
        img3.set_clim([0.0, 30.0])
        plt.savefig('results/x'+str(iSample)+'.png')




