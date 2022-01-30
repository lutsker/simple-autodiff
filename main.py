import autodiff as ad
import numpy as np


if __name__ == '__main__':
    print('Linear regression using SGD and self made autodiff')
    N = 1500
    D = 100.0
    alpha = -1.45
    beta = 2.2
    xx = np.arange(N) / float(N) * D
    yy = alpha * xx + beta + np.random.normal(loc=0, scale=0.125, size=N)

    # Model
    eta = 0.05 / N
    epochs = 500
    a = ad.Numtor(0.1)
    b = ad.Numtor(0.2)
    for ii in range(epochs):
        for sample_index in np.arange(N):
            x = ad.Numtor(xx[sample_index])
            y = ad.Numtor(yy[sample_index])
            yp = a*x + b
            loss = (yp - y) * (yp - y)
            loss.backward()
            a = ad.Numtor(a.value - eta * a.grad)
            b = ad.Numtor(b.value - eta * b.grad)
    print(loss.value, a.value, b.value)
