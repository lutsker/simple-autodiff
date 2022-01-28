from .numtor import Numtor
from .ops import *
import pytest


def test_exp():
    x = Numtor(3)
    a = Numtor(8)
    y = np_exp(a*x)
    y.backward()
    assert x.grad == y.value * a.value

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
