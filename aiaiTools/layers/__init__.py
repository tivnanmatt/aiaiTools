
import tensorflow as tf
import numpy as np
import aiaiTools
from aiaiTools.utils import Ones, Zeros

class BaseLayer(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(BaseLayer,self).__init__(**kwargs)
    def call(self,x):
        raise NotImplementedError

from . import linear
from . import shape
from . import spectral
from . import loss            
from . import meta



