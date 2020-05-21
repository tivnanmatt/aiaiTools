
import tensorflow as tf

import aiaiTools
from aiaiTools.utils import Ones, Zeros, Rand, Randn, Range

from ..ct import Nonlinear, CTReconstructor_NLPWLS, CTReconstructor_NLWLS, CTReconstructor_NLLS
from ..optimization import LeastSquares, WeightedLeastSquares, PenalizedWeightedLeastSquares
from aiaiTools.layers.meta import LinearSequence

class Polyenergetic(Nonlinear):
    def __init__(self,A,Q,S,B,G,x_shape,l_shape,y_shape, **kwargs):
        tf.keras.Model.__init__(self,**kwargs)
        self.A = A
        self.Q = Q
        self.S = S
        self.B = B
        self.G = G
        Nonlinear.__init__( self, 
                            LinearSequence(self.A,self.Q),
                            LinearSequence(self.S,self.B,self.G),
                            x_shape,
                            l_shape,
                            y_shape,
                            **kwargs)
        self.x_shape=x_shape
        self.l_shape=l_shape
        self.y_shape=y_shape
    def compute_output_shape(self,input_shape):
        return tf.TensorShape(self.y_shape)


class SpectralCTReconstructor_NLPWLS(CTReconstructor_NLPWLS):
    def __init__(self,y,A,Q,S,B,G,ySigmaInv,penaltyLoss,x_shape,l_shape,y_shape,**kwargs):
        tf.keras.Model.__init__(self,**kwargs)
        self.x_shape = x_shape
        self.l_shape = l_shape
        self.y_shape = y_shape
        self.ySigmaInv = ySigmaInv
        self.forwardModel = Polyenergetic(A,Q,S,B,G,x_shape,l_shape,y_shape,**kwargs)
        PenalizedWeightedLeastSquares.__init__(self,self.forwardModel,y,ySigmaInv,penaltyLoss,x_shape,**kwargs)

class SpectralCTReconstructor_NLWLS(SpectralCTReconstructor_NLPWLS):
    def __init__(self,y,A,Q,S,B,G,ySigmaInv,x_shape,l_shape,y_shape,**kwargs):
        penaltyLoss = aiaiTools.layers.loss.Null()
        SpectralCTReconstructor_NLPWLS.__init__(self,y,A,Q,S,B,G,ySigmaInv,penaltyLoss,x_shape,l_shape,y_shape,**kwargs)

class SpectralCTReconstructor_NLPLS(SpectralCTReconstructor_NLPWLS):
    def __init__(self,y,A,Q,S,B,G,penaltyLoss,x_shape,l_shape,y_shape,**kwargs):
        ySigmaInv = aiaiTools.layers.linear.Identity()
        SpectralCTReconstructor_NLPWLS.__init__(self,y,A,Q,S,B,G,ySigmaInv,penaltyLoss,x_shape,l_shape,y_shape,**kwargs)

class SpectralCTReconstructor_NLLS(SpectralCTReconstructor_NLWLS, SpectralCTReconstructor_NLPLS):
    def __init__(self,y,A,Q,S,B,G,x_shape,l_shape,y_shape,**kwargs):
        penaltyLoss = aiaiTools.layers.loss.Null()
        ySigmaInv = aiaiTools.layers.linear.Identity()
        SpectralCTReconstructor_NLPWLS.__init__(self,y,A,Q,S,B,G,ySigmaInv,penaltyLoss,x_shape,l_shape,y_shape,**kwargs)



