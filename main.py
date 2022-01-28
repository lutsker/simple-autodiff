import autodiff as ad
import numpy as np

if __name__ == '__main__':
    x = ad.Numtor(1)
    a = ad.Numtor(3)
    y = ad.np_exp(a * x)
    y.backward()
    print(x.grad)
    print(np.exp(3) * 3)
