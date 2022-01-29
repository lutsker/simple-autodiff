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

        args = ''
        if self.parents:
            for pp in self.parents:
                args += str(pp.value) + ' '
        else:
            args = str(value)
#        print('Numtor:', self.name, value, op, args)
 
                
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
    
    def backward_fixed(self, delta=1):
        self.grad = 1
        nodes = [self]
        while nodes:
#            print([(node.op, node) for node in nodes])
            current_node = nodes.pop(0)
            if current_node.op == 'exp':
                for parent in current_node.parents:
                    parent.grad += np_exp_grad(parent, current_node)
                    nodes.append(parent)
            if current_node.op == 'log':
                for parent in current_node.parents:
                    parent.grad += np_log_grad(parent, current_node)
                    nodes.append(parent)
            if current_node.op == 'add':
                for parent in current_node.parents:
                    parent.grad += np_sum_grad(parent, current_node)
                    if parent.op != 'assign':
                        nodes.append(parent)
            if current_node.op == 'mul':
                current_node.parents[0].grad += np_mul_grad(current_node.parents[1], current_node)
                current_node.parents[1].grad += np_mul_grad(current_node.parents[0], current_node)

                if current_node.parents[0].op != 'assign':
                    nodes.append(current_node.parents[0])
                if current_node.parents[0] is not current_node.parents[1]:
                    if current_node.parents[1].op != 'assign':
                        nodes.append(current_node.parents[1])
            if current_node.op == 'sigmoid':
                for parent in current_node.parents:
                    parent.grad += np_sigmoid_grad(parent, current_node)
                    nodes.append(parent)
            if current_node.op == 'sub':
                current_node.parents[0].grad += current_node.grad
                current_node.parents[1].grad += -current_node.grad
                nodes.append(current_node.parents[0])
                nodes.append(current_node.parents[1])
 #           print("Current node", current_node.name, current_node.op, current_node.grad)
