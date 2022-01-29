from .grad_ops import grad_ops


class Numtor:
    def __init__(self, value, op='assign', parents=None, name=None):
        self.op = op
        self.parents = None
        if parents is not None:
            self.parents = [*parents]
        self.value = value
        self.grad = 0
        self.name = name

        args = ''
        if self.parents:
            for pp in self.parents:
                args += str(pp.value) + ' '
        else:
            args = str(value)
        print('Numtor:', self.name, value, op, args)
 
                
    def __add__(self, other):
        return Numtor(self.value + other.value, op='add', parents=[self, other])

    def __sub__(self, other):
        return Numtor(self.value - other.value, op='sub', parents=[self, other])
    
    def __mul__(self, other):
        return Numtor(self.value * other.value, op='mul', parents=[self, other])
        
    def __str__(self):
        return 'value: {}'.format(self.value)
    
    def backward(self, delta=1):
        parent_ops = []
        if self.parents is not None:
            parent_ops=[parent.op for parent in self.parents]
        self.grad += delta
        print('Backprop:', self.name, self.op, delta, self.grad, parent_ops)
        if self.parents is not None:
            grad_ops[self.op](self, self.parents)
