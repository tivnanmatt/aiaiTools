

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
nMaterial=2
nPixel=182
nView=90
nEnergy=150
nChannel=3

x_shape=[nMaterial,nVoxelX,nVoxelY]
l_shape=[nEnergy,nPixel,nView]
y_shape=[nPixel,nView]

x0 = 2000e-6*shepp_logan_phantom()
x0 = resize(x0, [nVoxelX, nVoxelY])
x0 = tf.convert_to_tensor(x0,dtype=tf.float32)
x1 = 10e-6*aiaiTools.utils.RoiCircle2D(x_shape[1:], 50, 70, 10, dtype=tf.float32)
x0 = tf.expand_dims(x0,0)
x1 = tf.expand_dims(x1,0)
x = tf.concat([x0,x1],0)

theta = np.linspace(0.0,180.0,nView,endpoint=False,dtype=float)
def FP(x):
    return radon(x, theta=theta, circle=False)
def BP(x):
    return iradon(x, theta=theta, circle=False, filter=None, output_size=nVoxelX)
A = aiaiTools.layers.meta.LinearRepeat(aiaiTools.layers.linear.NumpyLinearOperator(FP,BP,[nVoxelX,nVoxelY],[nPixel,nView],dynamic=True),dynamic=True)

massAttenuationSpectra = np.zeros([nEnergy,nMaterial],dtype=float)
massAttenuationSpectra[:,0] = aiaiTools.spektr.mass_attenuation_spectrum([1,1,8])
massAttenuationSpectra[:,1] = aiaiTools.spektr.mass_attenuation_spectrum([53])
massAttenuationSpectra = tf.convert_to_tensor(massAttenuationSpectra,dtype=tf.float32)
Q = aiaiTools.layers.spectral.Q_Tensordot(massAttenuationSpectra,inputShape=[nMaterial,nPixel,nView],dynamic=True)

sourceSpectra1 = aiaiTools.spektr.sourceSpectrum(70.0).reshape([150,])
sourceSpectra2 = aiaiTools.spektr.sourceSpectrum(100.0).reshape([150,])
sourceSpectra3 = aiaiTools.spektr.sourceSpectrum(130.0).reshape([150,])

sourceSpectra1 = tf.convert_to_tensor(sourceSpectra1,dtype=tf.float32)
sourceSpectra2 = tf.convert_to_tensor(sourceSpectra2,dtype=tf.float32)
sourceSpectra3 = tf.convert_to_tensor(sourceSpectra3,dtype=tf.float32)

sourceSpectra1 = tf.reshape(sourceSpectra1, [1,nEnergy,1,1])
sourceSpectra2 = tf.reshape(sourceSpectra2, [1,nEnergy,1,1])
sourceSpectra3 = tf.reshape(sourceSpectra3, [1,nEnergy,1,1])

sourceSpectra = tf.concat([sourceSpectra1,sourceSpectra2,sourceSpectra3],axis=0)

preFilterSpectra = np.exp(-0.0027*10.0*aiaiTools.spektr.mass_attenuation_spectrum([13])).reshape([150,])
preFilterSpectra = tf.convert_to_tensor(preFilterSpectra,dtype=tf.float32)
preFilterSpectra = tf.reshape(preFilterSpectra, [1,nEnergy,1,1])

S0_A = aiaiTools.layers.shape.ExpandDims(inputShape=[nEnergy,nPixel,nView])
S0_B = aiaiTools.layers.shape.Tile([nChannel,1,1,1],inputShape=[1,nEnergy,nPixel,nView])
S0_C = aiaiTools.layers.spectral.S0_Scale(sourceSpectra,preFilterSpectra)
S0 = aiaiTools.layers.meta.LinearSequence(S0_A,S0_B,S0_C)

(interactionSpectra, conversionSpectra, gains) = aiaiTools.spektr.sensitivitySpectrum_CsI(L_CsI=0.55)

interactionSpectra = tf.convert_to_tensor(interactionSpectra, dtype=tf.float32)
interactionSpectra = tf.reshape(interactionSpectra, [nEnergy,1,1])
S1 = aiaiTools.layers.spectral.S1_Scale(interactionSpectra)

conversionSpectra = tf.convert_to_tensor(conversionSpectra, dtype=tf.float32)
conversionSpectra = tf.reshape(conversionSpectra,[nEnergy])
S2 = aiaiTools.layers.spectral.S2_Tensordot(conversionSpectra, axes=[[0],[1]], inputShape=[nChannel, nEnergy, nPixel, nView])

S = aiaiTools.layers.meta.LinearSequence(S0,S1,S2)

B = aiaiTools.layers.linear.Identity()

gains = tf.reshape(gains, [1,1,1])
G = aiaiTools.layers.spectral.G_Scale(gains)



# A = aiaiTools.layers.linear.Identity()
# Q = aiaiTools.layers.linear.Identity()
S = aiaiTools.layers.linear.Identity()
B = aiaiTools.layers.linear.Identity()
G = aiaiTools.layers.linear.Identity()
# l_shape=x_shape
y_shape=l_shape

myPolyenergetic = aiaiTools.models.spectral.Polyenergetic(A,Q,S,B,G,x_shape,l_shape,y_shape)
y = myPolyenergetic(x)
# y = Ones([nChannel,nPixel,nView])

myEstimator = aiaiTools.models.spectral.SpectralCTReconstructor_NLLS(y,A,Q,S,B,G,x_shape,l_shape,y_shape)
y = myEstimator.forwardModel(x)

# F = myEstimator.F
# tmp = Ones(x_shape)
# for layer in F.layers:
#     tmp = layer(tmp)
# F1 = tmp

# M = aiaiTools.layers.linear.Scale(1.0/tf.sqrt(F1))
# # M = aiaiTools.layers.linear.Scale(1.0/tf.sqrt(F(Ones(x_shape))))
# A = aiaiTools.layers.meta.LinearSequence(M,A,dynamic=True)


myEstimator = aiaiTools.models.spectral.SpectralCTReconstructor_NLLS(y,A,Q,S,B,G,x_shape,l_shape,y_shape)

nIter = 10
learning_rate = 2e6
myEstimator.compile(learning_rate=learning_rate)
myEstimator.x = x*0
myEstimator.fit(nIter=nIter)
# xi = M(myEstimator.x)








