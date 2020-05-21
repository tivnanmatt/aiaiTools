
import tensorflow as tf

import aiaiTools
from .. import BaseLayer
from ..linear import LinearOperator
from aiaiTools.utils import Ones, Zeros


class Batch(BaseLayer):
    def __init__(self, layer, output_shape=None,**kwargs):
        self.layer = layer
        BaseLayer.__init__(self,**kwargs)
    def build(self,input_shape):
        self.layer.build(input_shape[1:])
    def call(self,x):
        return tf.map_fn(self.layer,x)
    def compute_output_shape(self,input_shape):
        input_shape = [dim for dim in input_shape]
        output_shape = [input_shape[0]] 
        for dim in self.layer.compute_output_shape(input_shape[1:]):
            output_shape.append(dim)
        output_shape = tf.TensorShape(output_shape)
        return output_shape

class Repeat2(BaseLayer):
    def __init__(self, layer, **kwargs):
        self.layer = layer
        self.timeDistributedLayer=tf.keras.layers.TimeDistributed(self.layer)
        super(Repeat2,self).__init__(**kwargs)
    def build(self,input_shape):
        self.layer.build(input_shape[1:])
    def call(self,x):
        perm = range(len(x.shape))
        perm[0] = 1
        perm[1] = 0
        x = tf.expand_dims(x,axis=0)
        x = tf.transpose(x,perm)
        x = self.timeDistributedLayer(x)
        x = tf.transpose(x,perm)
        x = tf.reshape(x,x.shape[1:])
        return x
    def compute_output_shape(self,input_shape):
        input_shape = [dim for dim in input_shape]
        output_shape = [input_shape[0]]
        output_shape.append(self.layer.compute_output_shape(input_shape[1:]))
        return output_shape

class Repeat(BaseLayer):
    def __init__(self, layer, **kwargs):
        self.layer=layer
        super(Repeat,self).__init__(**kwargs)
    def build(self,input_shape):
        self.layer.build(input_shape[1:])
    def call(self,x):
        return tf.map_fn(self.layer.call, x)
    def compute_output_shape(self,input_shape):
        input_shape = [dim for dim in input_shape]
        output_shape = [input_shape[0]] 
        for dim in self.layer.compute_output_shape(input_shape[1:]):
            output_shape.append(dim)
        output_shape = tf.TensorShape(output_shape)
        return output_shape

class Sequence(BaseLayer):
    def __init__(self, *args, **kwargs):
        self.layers = args
        super(Sequence, self).__init__(**kwargs)
    def call(self,x):
        for layer in self.layers:
            print()
            print()
            print(type(layer))
            print(x.shape)
            print()
            print()
            x = layer(x)
        return x
    def compute_output_shape(self, input_shape):
        shape = input_shape
        for layer in self.layers:
            shape = layer.compute_output_shape(shape)
        output_shape = shape
        return output_shape

class ParallelSum(BaseLayer):
    def __init__(self, *args, **kwargs):
        self.layers = args
        super(ParallelSum, self).__init__(**kwargs)
    def call(self,x):
        y = 0
        for layer in self.layers:
            y = y + layer(x)
        return y
    def compute_output_shape(self, input_shape):
        input_shape = [(1 if dim is None else dim) for dim in input_shape]
        y = 0
        for layer in self.layers:
            y = y + Ones(layer.compute_output_shape(input_shape))
        return y.shape

class LinearRepeat(LinearOperator):
    def __init__(self, layer, **kwargs):
        self.layer=layer
        super(LinearRepeat,self).__init__(**kwargs)
    def build(self,input_shape):
        self.layer.build(input_shape[1:])
    def dot(self,x):
        return tf.map_fn(self.layer.dot, x)
    def Tdot(self,x):
        return tf.map_fn(self.layer.Tdot, x)
    def compute_output_shape(self,input_shape):
        input_shape = [dim for dim in input_shape]
        output_shape = [input_shape[0]] 
        for dim in self.layer.compute_output_shape(input_shape[1:]):
            output_shape.append(dim)
        output_shape = tf.TensorShape(output_shape)
        return output_shape

class LinearSequence(LinearOperator):
    def __init__(self, *args, dynamic=False,**kwargs):
        self.layers = [layer for layer in args]
        if not dynamic: 
            dynamic = any([layer.dynamic for layer in self.layers])
        super(LinearSequence, self).__init__(dynamic=dynamic,**kwargs)
    def build(self,input_shape):    
        for layer in self.layers:
            print()
            print()
            print(type(layer))
            print(input_shape)
            layer.build(input_shape)
            input_shape = layer.compute_output_shape(input_shape)
            print(input_shape)
            print()
            print() 
    def dot(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    def Tdot(self,x):
        self.layers.reverse()
        for layer in self.layers:
            try:
                x = layer.Tdot(x)
            except Exception as e:
                raise Exception('Error while handling layer of type: ', type(layer), ' ' + str(e))
        self.layers.reverse()
        return x
    def compute_output_shape(self, input_shape):
        shape = input_shape
        for layer in self.layers:
            shape = layer.compute_output_shape(shape)
        output_shape = shape
        return output_shape




