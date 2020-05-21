
import tensorflow as tf

import aiaiTools
from .. import BaseLayer
from aiaiTools.utils import Ones, Zeros


class BaseLoss(BaseLayer):
    def __init__(self, **kwargs):
        super(BaseLoss, self).__init__(**kwargs)
    def compute_output_shape(self, input_shape):
        return Zeros([1]).shape

class MultivariateGaussian(BaseLoss):
    def __init__(self, mu, SigmaInv,  **kwargs):
        self.mu = mu
        self.SigmaInv = SigmaInv
        super(MultivariateGaussian,self).__init__(**kwargs)    
    def call(self,x):
        return tf.reshape(0.5*tf.reduce_sum((x-self.mu)*self.SigmaInv(x-self.mu)),[1])

class QuadraticRegularizer(BaseLoss):
    def __init__(self, R, **kwargs):
        self.R = R
        super(QuadraticRegularizer,self).__init__(**kwargs)
    def call(self,x):
        return tf.reshape( tf.reduce_sum(x*self.R(x)) ,[1])

class Null(BaseLoss):
    def __init__(self,**kwargs):
        super(Null,self).__init__(**kwargs)    
    def call(self,x):
        return Zeros([1])

