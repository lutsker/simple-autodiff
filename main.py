import autodiff as ad
import numpy as np


if __name__ == '__main__':
    print('Linear regression using SGD and self made autodiff')
    N = 2000
    D = 100.0
    alpha = -1.45
    beta = 2.2
    xx = np.arange(N) / float(N) * D
    yy = alpha * xx + beta + np.random.normal(loc=0, scale=0.25, size=N)

    # Model
    eta = 0.00002
    epochs = 1800
    a = ad.Numtor(0.1)
    b = ad.Numtor(0.2)
    for _ in range(epochs):
        for sample_index in np.arange(N):
            x = ad.Numtor(xx[sample_index])
            y = ad.Numtor(yy[sample_index])
            yp1 = a*x + b
            w = yp1 - y
            loss = w * w
            loss.backward_fixed()
            a = ad.Numtor(a.value - eta * a.grad)
            b = ad.Numtor(b.value - eta * b.grad) 
    print(loss.value, a.value, b.value)
