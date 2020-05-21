
import tensorflow as tf

import aiaiTools
from aiaiTools.utils import Ones, Zeros, Rand, Randn, Range
from .. import BaseModel
from aiaiTools.layers.linear import Scale
from aiaiTools.layers.meta import LinearSequence
from ..optimization import LeastSquares, WeightedLeastSquares, PenalizedWeightedLeastSquares

class Linear(BaseModel):
    def __init__(   self,
                    AA,
                    x_shape,
                    y_shape,
                    **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.AA = AA
        self.Dx = Scale(Zeros(x_shape),trainable=False) 
        self.Dy = Scale(Zeros(y_shape),trainable=False)
    def call(self, x):
        self.x = x
        self.y = self.AA(self.x)
        return self.y
    def compute_output_shape(self, input_shape):
        return self.AA.compute_output_shape(input_shape)
    def _get_x(self): 
        return self.Dx.kernel
    def _set_x(self,val):
        self.Dx.kernel=val
    def _get_y(self): 
        return self.Dy.kernel
    def _set_y(self,val):
        self.Dy.kernel=val
    x = property(_get_x, _set_x)
    y = property(_get_y, _set_y)

class Nonlinear(BaseModel):
    def __init__(   self, 
                    AA, 
                    BB,
                    x_shape,
                    l_shape,
                    y_shape,
                    **kwargs):
        super(Nonlinear, self).__init__(**kwargs)
        self.AA = AA
        self.BB = BB
        self.Dx = Scale(Zeros(x_shape),trainable=False) 
        self.Dl = Scale(Zeros(l_shape),trainable=False) 
        self.Dz = Scale(Zeros(l_shape),trainable=False) 
        self.Dy = Scale(Zeros(y_shape),trainable=False) 
    def call(self, x):
        self.x = x
        self.l = self.AA(self.x)
        self.z = tf.math.exp(-self.l)
        self.y = self.BB(self.z)
        return self.y
    def _get_x(self): 
        return self.Dx.kernel
    def _set_x(self,val):
        self.Dx.kernel=val
    def _get_l(self): 
        return self.Dl.kernel
    def _set_l(self,val):
        self.Dl.kernel=val
    def _get_z(self): 
        return self.Dz.kernel
    def _set_z(self,val):
        self.Dz.kernel=val
    def _get_y(self): 
        return self.Dy.kernel
    def _set_y(self,val):
        self.Dy.kernel=val
    x = property(_get_x, _set_x)
    l = property(_get_l, _set_l)
    z = property(_get_z, _set_z)
    y = property(_get_y, _set_y)


class CTReconstructor_NLPWLS(PenalizedWeightedLeastSquares):
    def __init__(   self,y,AA,BB,ySigmaInv,penaltyLoss,x_shape,l_shape,y_shape,**kwargs):
        self.x_shape = x_shape
        self.l_shape = l_shape
        self.y_shape = y_shape
        self.ySigmaInv = ySigmaInv
        self.forwardModel = Nonlinear(AA,BB,x_shape,l_shape,y_shape,**kwargs)
        PenalizedWeightedLeastSquares.__init__(self,forwardModel,y,ySigmaInv,penaltyLoss,x_shape,**kwargs) 
    def _get_AA(self):
        return self.forwardModel.AA
    def _set_AA(self,val):
        self.forwardModel.AA = val
    def _get_BB(self):
        return self.forwardModel.BB
    def _set_BB(self,val):
        self.forwardModel.BB = val
    def _get_Dl(self):
        return self.forwardModel.Dl
    def _set_Dl(self,val):
        self.forwardModel.Dl = val
    def _get_Dz(self):
        return self.forwardModel.Dz
    def _set_Dz(self,val):
        self.forwardModel.Dz = val
    def _get_Dy(self):
        return self.forwardModel.Dy
    def _set_Dy(self,val):
        self.forwardModel.Dy = val
    def _get_F(self):
        return LinearSequence(self.AA,self.Dz,self.BB,self.ySigmaInv,self.BB.transpose(),self.Dz.transpose(),self.AA.transpose()) 
    AA = property(_get_AA, _set_AA)
    BB = property(_get_BB, _set_BB)
    Dl = property(_get_Dl, _set_Dl)
    Dz = property(_get_Dz, _set_Dz)
    Dy = property(_get_Dy, _set_Dy)
    F = property(_get_F)


class CTReconstructor_NLWLS(CTReconstructor_NLPWLS):
    def __init__(   self,y,AA,BB,ySigmaInv,x_shape,l_shape,y_shape,**kwargs):
        penaltyLoss = aiaiTools.layers.loss.Null()
        CTReconstructor_NLPWLS.__init__(self,y,AA,ySigmaInv,penaltyLoss,x_shape,l_shape,y_shape,**kwargs)

class CTReconstructor_NLPLS(CTReconstructor_NLPWLS):
    def __init__(   self,y,AA,BB,penaltyLoss,x_shape,l_shape,y_shape,**kwargs):
        ySigmaInv = aiaiTools.layers.linear.Identity()
        CTReconstructor_NLPWLS.__init__(self,y,AA,ySigmaInv,penaltyLoss,x_shape,l_shape,y_shape,**kwargs)

class CTReconstructor_NLLS(CTReconstructor_NLWLS, CTReconstructor_NLPLS):
    def __init__(   self,y,AA,BB,x_shape,l_shape,y_shape,**kwargs):
        penaltyLoss = aiaiTools.layers.loss.Null()
        ySigmaInv = aiaiTools.layers.linear.Identity()
        CTReconstructor_NLPWLS.__init__(self,y,AA,ySigmaInv,penaltyLoss,x_shape,l_shape,y_shape,**kwargs)




