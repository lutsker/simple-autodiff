from numtor import Numtor
from ops import *



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
