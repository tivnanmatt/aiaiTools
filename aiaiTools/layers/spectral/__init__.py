
import tensorflow as tf

import aiaiTools
from ..linear import Scale, Tensordot


class Q_Tensordot(Tensordot):
    def __init__(self,massAttenuationSpectra,axes=[[1],[0]],**kwargs):
        self.massAttenuationSpectra = massAttenuationSpectra
        if axes is None:
            axes = [[1],[0]]
        super(Q_Tensordot,self).__init__(self.massAttenuationSpectra, axes=axes,**kwargs)

class S0_Scale(Scale):
    def __init__(self,sourceSpectra,preFilterSpectra=None,**kwargs):
        if preFilterSpectra is None:
            preFilterSpectra = 1
        self.sourceSpectra = sourceSpectra
        self.preFilterSpectra = preFilterSpectra
        scaleSpectra = sourceSpectra*preFilterSpectra
        super(S0_Scale,self).__init__(scaleSpectra,**kwargs)

class S1_Scale(Scale):
    def __init__(self,interactionSpectra,**kwargs):
        self.interactionSpectra = interactionSpectra
        super(S1_Scale,self).__init__(self.interactionSpectra,**kwargs)

class S2_Tensordot(Tensordot):
    def __init__(self,conversionSpectra,axes=None,**kwargs):
        if axes is None:
            axes = [[1],[0]]
        self.conversionSpectra = conversionSpectra
        super(S2_Tensordot,self).__init__(self.conversionSpectra,axes=axes,**kwargs)

class G_Scale(Scale):
    def __init__(self,gains,**kwargs):
        self.gains = gains
        super(G_Scale,self).__init__(self.gains,**kwargs)

