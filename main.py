import numpy as np

class Numtor:
    def __init__(self, value, op='assign', parents=None):
        self.op = op
        self.parents = None
        if parents is not None:
            self.parents = [*parents]
        self.value = value
        self.grad = 0
        
 #       args = ''
 #       if self.parents:
 #           for pp in self.parents:
 #               args += str(pp.value) + ' '
 #       else:
 #           args = str(value)
        #print('Numtor:', op, args)
        
    def __add__(self, other):
        ## ToDo(vlut): the value shall be computed by refered op
        ##   return Numtor(op='add', parents=[self, other])
        ##   in Numtor constructor: 
        ##      self.value = ops[self.op](parents)
        ##   if it is a leaf node, then value shall be not None
        return Numtor(self.value + other.value, op='add', parents=[self, other])
    
    def __mul__(self, other):
        return Numtor(self.value * other.value, op='mul', parents=[self, other])
        
    def __str__(self):
        return 'value: {}'.format(self.value)
    
    def backward(self, delta=1):
#        parent_ops = []
#        if self.parents is not None:
#            parent_ops=[parent.op for parent in self.parents]
#        print(self.op, delta, parent_ops)
        self.grad += delta
        if self.parents is not None:
            # ToDo(vlut): use grad_ops[self.op](self.parents, self) instead
            if self.op == 'exp':
                for parent in self.parents:
                    parent.backward(np_exp_grad(parent, self))
            if self.op == 'log':
                for parent in self.parents:
                    parent.backward(np_log_grad(parent, self))
            if self.op == 'add':
                for parent in self.parents:
                    parent.backward(np_sum_grad(parent, self))
            if self.op == 'mul':
                self.parents[0].backward(np_mul_grad(self.parents[1], self))
                self.parents[1].backward(np_mul_grad(self.parents[0], self))
            if self.op == 'sigmoid':
                for parent in self.parents:
                    parent.backward(np_sigmoid_grad(parent, self))

def np_exp(arg: Numtor):
    return Numtor(np.exp(arg.value), op='exp', parents=[arg])

def np_log(arg: Numtor):
    return Numtor(np.log(arg.value), op='log', parents=[arg])

def np_sigmoid(arg: Numtor):
    return Numtor(1/(1+np.exp(-arg.value)), op='sigmoid', parents=[arg])

def np_sigmoid_grad(parent, this):
    return this.value * (1-this.value) * this.grad

def np_exp_grad(parent, this):
    return this.value * this.grad
    
def np_log_grad(parent, this):
    return 1/parent.value * this.grad
    
def np_sum_grad(parent, this):
    return this.grad

def np_mul_grad(parent, this):
    return parent.value * this.grad                   


if __name__ == '__main__':
    x = Numtor(25)
    y = Numtor(4)
    a = Numtor(125)
    b = Numtor(100)

    c = np_log(x*x*x + b*x + a*y*y)
    print('test')
    print(np.log(x.value**3+b.value*x.value+a.value*y.value**2))
    print((3*x.value*x.value+b.value) / (x.value**3 + b.value * x.value + a.value*y.value**2))
    print((2*a.value*y.value)/ (x.value**3 + b.value * x.value + a.value*y.value**2))

    print('autodif:')
    print(c.value)
    c.backward()
    print(x.grad)
    print(y.grad)
