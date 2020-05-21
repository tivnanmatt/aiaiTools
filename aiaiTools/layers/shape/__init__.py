
import tensorflow as tf
import aiaiTools
from ..linear import LinearOperator

class Reshape(LinearOperator):
    def __init__(self, target_shape, **kwargs):
        self.target_shape = target_shape
        super(Reshape,self).__init__(**kwargs)
    def dot(self,x):
        return tf.reshape(x,self.target_shape)
    def compute_output_shape(self,input_shape):
        return self.dot(aiaiTools.utils.Ones(input_shape)).shape

class ExpandDims(LinearOperator):
    def __init__(self,axis=0,**kwargs):
        self.axis = axis
        super(ExpandDims,self).__init__(**kwargs)
    def dot(self,x):
        return tf.expand_dims(x,self.axis)
    def compute_output_shape(self,input_shape):
        return self.dot(aiaiTools.utils.Ones(input_shape)).shape

class Permute(LinearOperator):
    def __init__(self,perm,**kwargs):
        self.perm = perm
        super(Permute,self).__init__(**kwargs)
    def dot(self,x):
        return tf.transpose(x,self.perm)
    def Tdot(self,x):
        rng = range(len(x.shape))
        Tperm = [self.perm.index(dim) for dim in range(len(x.shape))]
        return tf.transpose(x,Tperm)
    def compute_output_shape(self,input_shape):
        return [input_shape[dim] for dim in self.perm]


class Tile(LinearOperator):
    def __init__(self,multiples,**kwargs):
        self.multiples=multiples
        LinearOperator.__init__(self,**kwargs)
    def dot(self,x):
        return tf.tile(x,self.multiples)
    def compute_output_shape(self,input_shape):
        output_shape = [input_shape[ii]*self.multiples[ii] for ii in range(len(input_shape))]
        return tf.TensorShape(output_shape)



