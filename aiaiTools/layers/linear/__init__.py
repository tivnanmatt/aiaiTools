
import tensorflow as tf
import numpy as np

import aiaiTools
from .. import BaseLayer
from aiaiTools.utils import Ones, Zeros

class LinearOperator(BaseLayer):
    def __init__(self, inputShape=None,**kwargs):
        self.inputShape=inputShape
        super(LinearOperator,self).__init__(**kwargs)
    def call(self,x):
        return self.dot(x)
    def dot(self,x):
        raise NotImplementedError
    def Tdot(self,dy,x=None):
        if x is None:
            try:
                x = Ones(self.inputShape)
            except:
                pass
        if x is None:
            try:
                x = Ones(self.input_shape)
            except:
                pass
        if x is None:
            raise Exception('Error in Tdot. layer input shape is ambiguous.') 
        with tf.GradientTape() as g:
            g.watch(x)
            y = self.dot(x)
        return g.gradient(y,x,dy)
    def transpose(self):
        return TransposeLinearOperator(self)
    def compute_output_shape(self,input_shape):
        input_shape = [  ( 1 if dim is None else dim )  for dim in input_shape]
        x = Ones(input_shape)
        y = self.call(x)
        return y.shape
        



class TransposeLinearOperator(LinearOperator):
    def __init__(self, originalLayer, **kwargs):
        self.originalLayer = originalLayer
        super(TransposeLinearOperator, self).__init__(**kwargs)
    def dot(self,x):
        print("DDDDDD")
        print(type(self.originalLayer))
        print(type(x))
        return self.originalLayer.Tdot(x)
    def Tdot(self,x):
        return self.originalLayer.dot(x)


class Scale(LinearOperator):
    def __init__(self, kernel_init, **kwargs):
        self.kernel_init = kernel_init
        self.kernel = kernel_init
        super(Scale,self).__init__(**kwargs)
    def build(self, input_shape,dtype=tf.float32):
        self.kernel = self.add_weight(  shape=self.kernel_init.shape,
                                        initializer='zeros',
                                        trainable=True,
                                        dtype=dtype)
        self.set_weights([self.kernel_init])
    def dot(self,x):
        return self.kernel*x
    def Tdot(self,dy):
        return tf.math.conj(self.kernel)*dy
    def compute_output_shape(self,input_shape):
        input_shape = [  ( 1 if dim is None else dim )  for dim in input_shape]
        x = Ones(input_shape)
        y = self.call(x)
        return y.shape
    


class Identity(LinearOperator):
    def __init__(self, **kwargs):
        super(Identity,self).__init__(**kwargs)
    def dot(self,x):
        return x
    def Tdot(self,dy,x=None):
        return dy

class MatrixMultiply(LinearOperator):
    def __init__(self, kernel_init, **kwargs):
        self.kernel_init = kernel_init
        self.kernel = kernel_init
        super(MatrixMultiply,self).__init__(**kwargs)
    def build(self, input_shape):
        self.kernel = self.add_weight(  shape=self.kernel_init.shape,
                                        initializer='zeros',
                                        trainable=True)
        self.set_weights([self.kernel_init])
    def dot(self,x):
        return tf.matmul(self.kernel,x)
    def Tdot(self,dy,x=None):
        return tf.matmul(tf.transpose(self.kernel),dy)

class Tensordot(LinearOperator):
    def __init__(self, kernel_init, axes=None,**kwargs):
        self.kernel_init = kernel_init
        self.kernel = kernel_init
        self.axes = axes
        super(Tensordot,self).__init__(**kwargs)
    def build(self,input_shape):
        self.inputShape=input_shape
        self.kernel = self.add_weight(  shape=self.kernel_init.shape,
                                        initializer='zeros',
                                        trainable=True)
        self.set_weights([self.kernel_init])
    def dot(self,x):
        return tf.tensordot(self.kernel,x,self.axes)
    def compute_output_shape(self,input_shape):
        input_shape = input_shape.as_list()
        kernel_shape = self.kernel.shape.as_list()
        for dim in self.axes[0]:
            kernel_shape.pop(dim)
        for dim in self.axes[1]:
            input_shape.pop(dim)
        return tf.TensorShape(kernel_shape + input_shape)

class NumpyLinearOperator(LinearOperator):
    def __init__(self, dotFun, TdotFun, x_shape, y_shape, **kwargs):
        def tf_dotFun(x):
            x = x.numpy().astype(float)
            y = dotFun(x)
            y = tf.convert_to_tensor(y, dtype=tf.float32)
            return y 
        def tf_TdotFun(x):
            x = x.numpy().astype(float)
            y = TdotFun(x)
            y = tf.convert_to_tensor(y, dtype=tf.float32)
            return y
        @tf.custom_gradient
        def dot_fun(x):
            y = tf.py_function(tf_dotFun, [x], tf.float32)
            def grad(dy):
                dx = tf.py_function(tf_TdotFun, [dy], tf.float32)
                return dx
            return (y, grad)
        @tf.custom_gradient
        def Tdot_fun(x):
            y = tf.py_function(tf_TdotFun, [x], tf.float32)
            def grad(dy):
                dx = tf.py_function(tf_dotFun, [dy], tf.float32)
                return dx
            return (y, grad)
        self.dot_fun = dot_fun
        self.Tdot_fun = Tdot_fun
        self.x_shape = x_shape
        self.y_shape = y_shape
        LinearOperator.__init__(self,inputShape=self.x_shape)
    def dot(self,x):
        return self.dot_fun(x)
    def Tdot(self,dy):
        return self.Tdot_fun(dy)
    def compute_output_shape(self, input_shape):
        y = Ones(self.y_shape)
        output_shape = y.shape
        return y.shape


class FourierTransform2D(LinearOperator):
    def __init__(self,fft_length=None,**kwargs): 
        self.fft_length=fft_length
        super(FourierTransform2D,self).__init__(**kwargs)
    def dot(self,x):
        return tf.signal.rfft2d(x,fft_length=self.fft_length)
    def Tdot(self,x):
        return tf.signal.irfft2d(x,fft_length=self.fft_length)

class InverseFourierTransform2D(LinearOperator):
    def __init__(self,fft_length=None,**kwargs):
        self.fft_length=fft_length
        super(InverseFourierTransform2D,self).__init__(**kwargs)
    def dot(self,x):
        return tf.signal.irfft2d(x,fft_length=self.fft_length)
    def Tdot(self,x):
        return tf.signal.rfft2d(x,fft_length=self.fft_length)

class FourierOperator2D(LinearOperator):
    def __init__(self,h,fft_length=None,frequencyDomain=True,**kwargs):
        if fft_length is None:
            fft_length = h.shape
        self.fft_length = fft_length
        self.FT = FourierTransform2D(fft_length=fft_length)
        self.IFT = InverseFourierTransform2D(fft_length=fft_length)
        self.frequencyDomain=frequencyDomain
        if self.frequencyDomain:
            self.kernel = self.FT(h)
        else:
            self.kernel = h
        super(FourierOperator2D, self).__init__(**kwargs)
    def build(self, input_shape,dtype=tf.float32):
        self.fft_length=input_shape
        self.FT = FourierTransform2D(fft_length=self.fft_length)
        self.IFT = InverseFourierTransform2D(fft_length=self.fft_length)
        if tf.executing_eagerly():
            kernel_init = self.kernel.numpy()
        else:
            kernel_init = self.kernel.eval()
        self.kernel = self.add_weight(  shape=self.kernel.shape,
                                        initializer=tf.keras.initializers.Constant(kernel_init),
                                        trainable=True,
                                        dtype=dtype)
    def dot(self,x):
        x = self.FT.dot(x)
        x = tf.cast(x,dtype=tf.complex64)
        x = self.H*x
        x = self.IFT.dot(x)
        return x
    def Tdot(self,x):
        x = self.FT.dot(x)
        x = tf.math.conj(H)*x
        x = self.IFT.dot(x)
        return x
    def compute_output_shape(self,input_shape):
        return input_shape
    def _set_H(self,val):
        if self.frequencyDomain:
            self.kernel.assign(val)
        else:
            self.h = self.IFT(val)
    def _get_H(self):
        if self.frequencyDomain:
            H = self.kernel
        else:
            H = self.FT(self.kernel)
        return tf.cast(H,dtype=tf.complex64)
    def _set_h(self,val):
        if self.frequencyDomain:
            self.H = self.FT(val)
        else:
            self.kernel.assign(val)
    def _get_h(self):
        if self.frequencyDomain:
            h = self.IFT(self.kernel)
        else:
            h = self.kernel
        return tf.cast(h,dtype=tf.float32)
    H = property(_get_H, _set_H)
    h = property(_get_h, _set_h)

class FourierApproximation2D(FourierOperator2D):
    def __init__(self, originalLayer, wTest, symmetric=False,**kwargs):
        fft_length = wTest.shape
        self.FT = FourierTransform2D(fft_length=fft_length)
        self.IFT = InverseFourierTransform2D(fft_length=fft_length)
        layerFrequencyResponse = self.FT(originalLayer(wTest))
        testFrequencyResponse = self.FT(wTest)
        H = layerFrequencyResponse/testFrequencyResponse
        if symmetric:
            H = tf.math.abs(H)
        h = self.IFT(H)
        super(FourierApproximation2D,self).__init__(h,fft_length)

class FourierConvolution2D(LinearOperator):
    def __init__(self, h, fourier=False,**kwargs):
        self.h = h
        def dotFourier(x):
            x = tf.pad(x,[[0, self.h.shape[0]-1],[0, self.h.shape[1]-1]],'CONSTANT')
            H = tf.signal.rfft2d(self.h, fft_length=x.shape)
            X = tf.signal.rfft2d(x, fft_length=x.shape)
            Y = X*H
            y = tf.signal.irfft2d(Y, fft_length=x.shape)
            shift0 = -tf.cast(tf.math.floor((self.h.shape[0]-1)/2),dtype=tf.int32)
            shift1 = -tf.cast(tf.math.floor((self.h.shape[1]-1)/2),dtype=tf.int32)
            y = tf.roll(y, shift=[shift0,shift1], axis=[0,1])
            y = y[0:y.shape[0]-self.h.shape[0]+1, 0:y.shape[1]-self.h.shape[1]+1]
            return y
        self.dotLayer = tf.keras.layers.Lambda(function=dotFourier)
        super(FourierConvolution2D, self).__init__(**kwargs)
    def dot(self,x):
        return self.dotLayer(x)


class Convolution2D(LinearOperator):
    def __init__(self, h,**kwargs):
        self.h = h
        def dotDirect(x):
            h = self.h
            h = tf.reshape(h, [h.shape[0], h.shape[1],1,1])
            h = tf.reverse(h,[0,1])
            padding = [[0,0],[0,0]]
            padding[0][0] = tf.cast(tf.math.ceil( (h.shape[0]-1)/2 ),dtype=tf.int32)
            padding[0][1] = tf.cast(tf.math.floor( (h.shape[0]-1)/2 ),dtype=tf.int32)
            padding[1][0] = tf.cast(tf.math.ceil( (h.shape[1]-1)/2 ),dtype=tf.int32)
            padding[1][1] = tf.cast(tf.math.floor( (h.shape[1]-1)/2 ),dtype=tf.int32)
            x = tf.pad(x,padding,'CONSTANT')
            x = tf.reshape(x, [1,x.shape[0],x.shape[1],1])
            y = tf.nn.conv2d(x, h, strides=1, padding='SAME')
            y = tf.reshape(y, [y.shape[1], y.shape[2]])
            y = y[padding[0][1]:padding[0][1]+y.shape[0]-h.shape[0]+1, padding[1][1]:padding[1][1]+y.shape[1]-h.shape[1]+1]
            return y
        self.dotLayer = tf.keras.layers.Lambda(function=dotDirect)
        super(Convolution2D, self).__init__(**kwargs)
    def dot(self,x):
        return self.dotLayer(x)


class RadiallySymmetricFilter2D(FourierOperator2D):
    def __init__(self, radiusFunction, kernel_shape, **kwargs): 
        xGrid = tf.range(kernel_shape[1],dtype=tf.float32)
        yGrid = tf.range(kernel_shape[0],dtype=tf.float32)
        xGrid = xGrid - xGrid[tf.cast(tf.math.floor((kernel_shape[1]-1)/2),dtype=tf.int32)]
        yGrid = yGrid - yGrid[tf.cast(tf.math.floor((kernel_shape[0]-1)/2),dtype=tf.int32)]
        xGrid, yGrid = tf.meshgrid(xGrid, yGrid)
        r = tf.math.sqrt(xGrid*xGrid + yGrid*yGrid)
        h = radiusFunction(r)
        fft_length = kernel_shape
        super(RadiallySymmetricFilter2D,self).__init__(h, fft_length, **kwargs) 


class GaussianBlur2D(RadiallySymmetricFilter2D):
    def __init__(self, sigma, kernel_shape,**kwargs):
        Sigma = sigma*sigma*tf.eye(2)
        detSigma = tf.linalg.det(Sigma)
        def gaussKernel(r):
            h = tf.math.exp(-0.5*r*r/sigma/sigma)
            h = h/2
            h = h/3.14159265359
            h = h/tf.math.sqrt(detSigma)
            return h
        super(GaussianBlur2D,self).__init__(gaussKernel, kernel_shape,**kwargs)






class FourierTransform1D(LinearOperator):
    def __init__(self,fft_length=None,**kwargs): 
        self.fft_length=fft_length
        super(FourierTransform1D,self).__init__(**kwargs)
    def dot(self,x):
        return tf.signal.rfft(x,fft_length=self.fft_length)
    def Tdot(self,x):
        return tf.signal.irfft(x,fft_length=self.fft_length)

class InverseFourierTransform1D(LinearOperator):
    def __init__(self,fft_length=None,**kwargs):
        self.fft_length=fft_length
        super(InverseFourierTransform1D,self).__init__(**kwargs)
    def dot(self,x):
        return tf.signal.irfft(x,fft_length=self.fft_length)
    def Tdot(self,x):
        return tf.signal.rfft(x,fft_length=self.fft_length)

class FourierOperator1D(LinearOperator):
    def __init__(self,h,fft_length=None,**kwargs):
        if fft_length is None:
            fft_length = h.shape
        self.fft_length = fft_length
        self._h = h
        self.FT = FourierTransform1D(fft_length=fft_length)
        self.IFT = InverseFourierTransform1D(fft_length=fft_length)
        self.kernel_init = self.FT(h)
        self.kernel = self.kernel_init
        super(FourierOperator1D, self).__init__(**kwargs)
    def build(self, input_shape,dtype=tf.float32):
        self.FT = FourierTransform1D(fft_length=input_shape)
        self.IFT = InverseFourierTransform1D(fft_length=input_shape)
        self.kernel_init = self.FT(self._h)
        self.kernel = self.kernel_init
        self.kernel = self.add_weight(  shape=self.kernel_init.shape,
                                        initializer='zeros',
                                        trainable=True,
                                        dtype=dtype)
        self.set_weights([self.kernel_init])
    def dot(self,x):
        x = self.FT.dot(x)
        x = tf.cast(x,dtype=tf.complex64)
        H = tf.cast(self.kernel,dtype=tf.complex64)
        x = H*x
        x = self.IFT.dot(x)
        return x
    def Tdot(self,x):
        x = self.IFT.Tdot(x)
        x = self.DH.Tdot(x)
        x = self.FT.Tdot(x)
        return x
    def compute_output_shape(self,input_shape):
        return input_shape
    def _set_H(self,val):
        self.kernel.assign(val)
    def _get_H(self):
        return self.kernel
    def _set_h(self,val):
        self.H = self.FT(val)
    def _get_h(self):
        return self.IFT(self.H)
    H = property(_get_H, _set_H)
    h = property(_get_h, _set_h)

