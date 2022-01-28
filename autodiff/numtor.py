from .grad_ops import grad_ops


class Numtor:
    def __init__(self, value, op='assign', parents=None):
        self.op = op
        self.parents = None
        if parents is not None:
            self.parents = [*parents]
        self.value = value
        self.grad = 0
                
    def __add__(self, other):
        return Numtor(self.value + other.value, op='add', parents=[self, other])
    
    def __mul__(self, other):
        return Numtor(self.value * other.value, op='mul', parents=[self, other])
        
    def __str__(self):
        return 'value: {}'.format(self.value)
    
    def backward(self, delta=1):
        self.grad += delta
        if self.parents is not None:
            grad_ops[self.op](self, self.parents)
