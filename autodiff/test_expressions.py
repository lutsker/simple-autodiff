from .numtor import Numtor
from .ops import *
import pytest


def test_exp():
    x = Numtor(3)
    a = Numtor(8)
    y = np_exp(a*x)
    y.backward()
    assert x.grad == pytest.approx(y.value * a.value)

def test_complex_log():
    x = Numtor(25)
    y = Numtor(4)
    a = Numtor(125)
    b = Numtor(100)

    c = np_log(x*x*x + b*x + a*y*y)
    assert c.value == pytest.approx(np.log(x.value**3+b.value*x.value+a.value*y.value**2))
    c.backward()
    assert x.grad == pytest.approx((3*x.value*x.value+b.value) / (x.value**3 + b.value * x.value + a.value*y.value**2))
    assert y.grad == pytest.approx((2*a.value*y.value)/ (x.value**3 + b.value * x.value + a.value*y.value**2))

def test_sigmoid():
    x1 = Numtor(1)
    x2 = Numtor(0)

    w11 = Numtor(0.1)
    w12 = Numtor(-0.2)
    w21 = Numtor(0.15)
    w22 = Numtor(-0.32)

    z1 = w11 * x1 + w12 * x2
    z2 = w21 * x1 + w22 * x2

    y = np_sigmoid(z1) + np_sigmoid(z2)
    y.backward()

    zz1 = w11.value * x1.value + w12.value * x2.value
    zz2 = w21.value * x1.value + w22.value * x2.value
    ss1 = 1/(1+np.exp(-zz1))
    ss2 = 1/(1+np.exp(-zz2))

    assert x2.grad == pytest.approx(ss1 * (1-ss1) * w12.value + ss2 * (1-ss2) * w22.value)
    assert x1.grad == pytest.approx(ss1 * (1-ss1) * w11.value + ss2 * (1-ss2) * w21.value)

def test_sub():
    a = Numtor(2)
    b = Numtor(5)
    x = a - b
    x.backward() 
    assert a.grad == pytest.approx(1)
    assert b.grad == pytest.approx(-1)

def test_sub_complex():
    a = Numtor(3)
    b = Numtor(-4)
    x = Numtor(9)
    z = np_exp(a*x - b)
    z.backward()
    assert a.grad == pytest.approx(z.value * x.value)
    assert b.grad == pytest.approx(-z.value)

def test_square_loss():
    a = Numtor(0.1)
    x = Numtor(2)
    z = a*x*a*x
    z.backward()
    assert a.grad == pytest.approx(2*a.value * x.value * x.value )
    assert x.grad == pytest.approx(2*x.value * a.value**2)

    x1 = Numtor(-3, name='x1')
    x2 = Numtor(2, name='x2')
    b = Numtor(-1, name='b')
    a = Numtor(4, name='a')
    y1 = x1 * b 
    z1 = x2 + y1
    w1 = a - z1
    l = w1 * w1  
    l.backward()
    assert x1.grad == pytest.approx(2*(a.value - x2.value - x1.value * b.value) * (-b.value) )
    assert x2.grad == pytest.approx(-2*(a.value - x2.value - x1.value * b.value) )

def test_square():
    a = Numtor(1, name='a')
    x = Numtor(2, name='x')
    c = Numtor(2, name='c')
    y = x * c
    b = (a+y)*(a+y)
    b.backward()
    print(x.grad, 2 * 5 * 2)
    assert x.grad == pytest.approx(20)
    assert c.grad == pytest.approx(20)

def test_chained():
    x1 = Numtor(1)
    x2 = Numtor(2)
    w1 = Numtor(1)
    w2 = Numtor(-1)
    w3 = Numtor(2)
    z = np_sigmoid( x1 * w1 + x2 * w2 )
    y = w3 * z
    y.backward()
    # dy/dw2 = w3 z(1-z)x1
    assert pytest.approx(w3.grad) == z.value
    assert pytest.approx(w1.grad) == w3.value * z.value * (1-z.value) * x1.value
    assert pytest.approx(w2.grad) == w3.value * z.value * (1-z.value) * x2.value

@pytest.mark.parametrize('x1val', [ -2.9, 0, 7.5])
@pytest.mark.parametrize('x2val', [ -3.4, 0, 1.1])
@pytest.mark.parametrize('wval', [ -1.2, 0, 5.6])
def test_chained_shared_weight(x1val, x2val, wval):
    x1 = Numtor(x1val)
    x2 = Numtor(x2val)
    w  = Numtor(wval)
    z = np_sigmoid( x1 * w + x2 * w )
    y = w * z
    y.backward()
    # dy/dw = z + w z(1-z)(x1 + x2)
    assert pytest.approx(w.grad) == z.value + w.value * z.value * (1-z.value) * (x1.value + x2.value)
