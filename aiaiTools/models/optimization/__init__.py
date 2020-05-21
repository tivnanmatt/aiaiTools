
import tensorflow as tf

import aiaiTools
from .. import BaseModel
from aiaiTools.utils import Ones, Zeros, Rand, Randn, Range
from aiaiTools.layers.linear import Scale
from aiaiTools.layers.meta import Sequence, ParallelSum

class Estimator(BaseModel):
    def __init__(   self, 
                    objectiveFunction,
                    x_shape,
                    dynamic=False,
                    **kwargs ):
        super(Estimator, self).__init__(**kwargs)
        self.Dx = Scale(Zeros(x_shape),name='Dx',dynamic=dynamic)
        self.Dx.build([1]) 
        self.objectiveFunction = objectiveFunction
        self.objectiveFunction.trainable=False
        inputs = tf.keras.Input(shape=[1])
        outputs = self.call(inputs)
        super(Estimator, self).__init__(inputs=inputs, outputs=outputs, **kwargs)
    def call(self, x, training=None):
        xsave = self.x
        if not training:
            self.x = self.x*0 + 1
        x = self.Dx(x)
        self.x = xsave
        return self.objectiveFunction(x)     
    def compile(self, optimizer=None, loss=None, learning_rate=1.0, **kwargs):
        if loss is None:
            loss = 'mean_absolute_error'
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        super(Estimator, self).compile(optimizer,loss,**kwargs)
    def fit(self, nIter=1, **kwargs):
        x = tf.ones([nIter],dtype=tf.float32)
        y = tf.zeros([nIter],dtype=tf.float32)
        kwargs.pop('x',None)
        kwargs.pop('y',None)
        kwargs.pop('batch_size',None)
        kwargs.pop('epochs', None)
        super(Estimator, self).fit(x=x, y=y, batch_size=1, epochs=1,**kwargs)
    def _get_x(self): 
        return self.Dx.kernel
    def _set_x(self,val):
        self.Dx.kernel.assign(val)
    x = property(_get_x, _set_x)

class MaximumAPosteriori(Estimator):
    def __init__(   self,
                    forwardModel,
                    likelihoodLoss,
                    priorLoss,                    
                    x_shape,
                    **kwargs):
        objectiveFunction = ParallelSum(Sequence(forwardModel, likelihoodLoss), priorLoss)
        super(MaximumAPosteriori,self).__init__(objectiveFunction, x_shape, **kwargs)

class MaximumLikelihood(MaximumAPosteriori):
    def __init__(   self,
                    forwardModel,
                    likelihoodLoss,
                    x_shape,
                    **kwargs):
        priorloss = aiaitools.layers.loss.null()
        super(MaximumLikelihood,self).__init__(forwardModel, likelihoodLoss, priorLoss, **kwargs)

class PenalizedWeightedLeastSquares(MaximumAPosteriori):
    def __init__( self, forwardModel, y, ySigmaInv, penaltyLoss, x_shape, **kwargs):
        y_shape = y.shape
        likelihoodLoss = aiaiTools.layers.loss.MultivariateGaussian(y,ySigmaInv)
        priorLoss = penaltyLoss
        super(PenalizedWeightedLeastSquares,self).__init__(forwardModel,likelihoodLoss,priorLoss,x_shape, **kwargs)

class WeightedLeastSquares(PenalizedWeightedLeastSquares):
    def __init__( self, forwardModel, y, ySigmaInv, x_shape, **kwargs):
        penaltyLoss = aiaiTools.layers.loss.Null()
        super(WeightedLeastSquares,self).__init__(forwardModel, y, ySigmaInv,penaltyLoss,x_shape, **kwargs)
 
class LeastSquares(WeightedLeastSquares):
    def __init__( self, forwardModel, y, x_shape,**kwargs):
        ySigmaInv = aiaiTools.layers.linear.Identity()
        super(LeastSquares,self).__init__(forwardModel, y, ySigmaInv, x_shape, **kwargs)

