def exp_grad(self, parents):
    for parent in parents:
        parent.backward(np_exp_grad(parent, self))

def log_grad(self, parents):
    for parent in parents:
        parent.backward(np_log_grad(parent, self))

def add_grad(self, parents):
    for parent in parents:
        parent.backward(np_sum_grad(parent, self))

def mul_grad(self, parents):
    parents[0].backward(np_mul_grad(parents[1], self))
    parents[1].backward(np_mul_grad(parents[0], self))

def sigmoid_grad(self, parents):
    for parent in parents:
        parent.backward(np_sigmoid_grad(parent, self))

def sub_grad(self, parents):
    parents[0].backward(self.grad)
    parents[1].backward(-self.grad)


grad_ops = {'exp': exp_grad, 'log': log_grad, 'add': add_grad, 'mul': mul_grad, 'sigmoid': sigmoid_grad,
'sub': sub_grad}


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
