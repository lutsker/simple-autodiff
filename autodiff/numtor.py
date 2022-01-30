from .grad_ops import *


class Numtor:
    def __init__(self, value, op='assign', parents=None, name=None):
        self.op = op
        self.parents = None
        if parents is not None:
            self.parents = [*parents]
        self.value = value
        self.grad = 0
        self.name = name
 
                
    def __add__(self, other):
        return Numtor(self.value + other.value, op='add', parents=[self, other])

    def __sub__(self, other):
        return Numtor(self.value - other.value, op='sub', parents=[self, other])
    
    def __mul__(self, other):
        return Numtor(self.value * other.value, op='mul', parents=[self, other])
        
    def __str__(self):
        return 'Numtor: {}'.format(self.value)
    
    def backward(self, delta=1):
        if self.parents is not None:
            self.grad = delta
            grad_ops[self.op](self, self.parents)
        else:
            self.grad += delta
