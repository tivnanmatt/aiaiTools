
import tensorflow as tf
import numpy as np

import aiaiTools


def Ones(shape, dtype=tf.float32):
    return tf.ones(shape, dtype=dtype)

def Zeros(shape, dtype=tf.float32):
    return tf.zeros(shape, dtype=dtype)

def Rand(shape, dtype=tf.float32):
    return tf.random.uniform(shape, dtype=dtype)

def Randn(shape, dtype=tf.float32):
    return tf.random.normal(shape, dtype=dtype)

def Range(shape, dtype=tf.float32):
    numel = tf.reduce_prod(shape)
    numel = tf.cast(numel, tf.int32)
    rng = tf.range(numel)
    rng = tf.cast(rng, dtype=dtype)
    rng = tf.reshape(rng, shape) 
    return rng

def RowColGrid(shape, dtype=tf.float32):
    rows = tf.range(shape[0],dtype=dtype)
    cols = tf.range(shape[1],dtype=dtype)
    return tf.meshgrid(rows,cols,indexing='ij')

def Dist(shape, row, col, dtype=tf.float32):
    rowGrid, colGrid = RowColGrid(shape,dtype=dtype)
    rowGrid = rowGrid - row
    colGrid = colGrid - col
    r2 = rowGrid*rowGrid + colGrid*colGrid
    r = tf.sqrt(r2)
    return r

def RoiCircle2D(shape, row, col, radius, dtype=tf.float32):
    r = Dist(shape, row, col)
    roi = r < radius
    roi = tf.cast(roi, dtype=dtype)
    return roi

def Impulse2D(shape, center=True, dtype=tf.float32):
    impulse = np.zeros(shape, dtype=float)
    if center:
        impulse[np.floor((shape[0]-1)/2).astype(int), np.floor((shape[1]-1)/2).astype(int)] = 1.0
    else:
        impulse[0,0] = 1
    impulse = tf.convert_to_tensor(impulse, dtype=dtype)
    return impulse

def Impulse1D(shape, center=True, dtype=tf.float32):
    impulse = np.zeros(shape, dtype=float)
    if center:
        impulse[np.floor((shape[0]-1)/2).astype(int)] = 1.0
    else:
        impulse[0] = 1
    impulse = tf.convert_to_tensor(impulse, dtype=dtype)
    return impulse

