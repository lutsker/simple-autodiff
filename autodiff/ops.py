from .numtor import Numtor
import numpy as np

def np_exp(arg: Numtor):
    return Numtor(np.exp(arg.value), op='exp', parents=[arg])

def np_log(arg: Numtor):
    return Numtor(np.log(arg.value), op='log', parents=[arg])

def np_sigmoid(arg: Numtor):
    return Numtor(1/(1+np.exp(-arg.value)), op='sigmoid', parents=[arg])
