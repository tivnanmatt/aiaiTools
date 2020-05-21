
import tensorflow as tf

import aiaiTools
from aiaiTools.utils import Ones, Zeros

class BaseModel(tf.keras.Model, aiaiTools.layers.BaseLayer):
    def __init__(self,**kwargs):
        self._dynamic=False
        super(BaseModel,self).__init__(**kwargs)
    def call(self,x):
        raise NotImplementedError
 

from . import optimization
from . import ct
from . import spectral



