

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


nVoxelX=128
nVoxelY=128
nPixel=182
nView=360

x_shape=[nVoxelX,nVoxelY]
y_shape=[nPixel,nView]

theta = np.linspace(0.0,180.0,nView,endpoint=False,dtype=float)
def FP(x):
    return radon(x, theta=theta, circle=False)
def BP(x):
    return iradon(x, theta=theta, circle=False, filter=None, output_size=nVoxelX)


x = shepp_logan_phantom()
x = resize(x, [nVoxelX, nVoxelY])
x = tf.convert_to_tensor(x,dtype=tf.float32)

A = aiaiTools.layers.linear.NumpyLinearOperator(FP,BP,x_shape,y_shape,dynamic=True)
y = A(x)

nIter = 1000
learning_rate = 1e-1

A = aiaiTools.layers.linear.NumpyLinearOperator(FP,BP,x_shape,y_shape,dynamic=True)
myForwardModel = aiaiTools.models.ct.Linear(A,x_shape,y_shape)
myEstimator = aiaiTools.models.optimization.LeastSquares(myForwardModel,y,x_shape,dynamic=True)
myEstimator.compile(learning_rate=learning_rate)
myEstimator.fit(nIter=nIter)

xi = myEstimator.x.numpy()
plt.figure()
img = plt.imshow(xi)
img.set_clim(0.0,1.0)
plt.savefig('results/LLS.png')


A = aiaiTools.layers.linear.NumpyLinearOperator(FP,BP,x_shape,y_shape)
ATA = aiaiTools.layers.meta.LinearSequence(A, A.transpose())
ATA_fourier = aiaiTools.layers.linear.FourierApproximation2D(ATA, x)
ATA_inv = aiaiTools.layers.linear.FourierOperator2D(ATA_fourier.IFT(1/ATA_fourier.H))
appodization = aiaiTools.layers.linear.GaussianBlur2D(0.5, x_shape)
M = aiaiTools.layers.linear.FourierOperator2D(ATA_fourier.IFT(1/tf.sqrt(ATA_fourier.H)*appodization.H))
# M = aiaiTools.layers.linear.FourierOperator2D(ATA_fourier.IFT(1/tf.sqrt(ATA_fourier.H)))
myForwardModel = aiaiTools.layers.meta.LinearSequence(M,A)
myEstimator = aiaiTools.models.optimization.LeastSquares(myForwardModel,y,x_shape,dynamic=True)
myEstimator.compile(learning_rate=learning_rate)
myEstimator.fit(nIter=nIter)

xi = M(myEstimator.x.numpy())
plt.figure()
img = plt.imshow(xi)
img.set_clim(0.0,1.0)
plt.savefig('results/PLLS.png')

