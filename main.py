import autodiff as ad
import numpy as np


if __name__ == '__main__':
    print('Linear regression using SGD and self made autodiff')
    N = 2000
    D = 100.0
    alpha = -1.45
    beta = 2.2
    xx = np.arange(N) / float(N) * D
    yy = alpha * xx + beta + np.random.normal(loc=0, scale=0.5, size=N)

    # Model
    eta = 0.00003
    epochs = 500
    a = ad.Numtor(0.1)
    b = ad.Numtor(0.2)
    for ii in range(epochs):
        for sample_index in np.arange(N):
            x = ad.Numtor(xx[sample_index])
            y = ad.Numtor(yy[sample_index])
            z1 = a*x 
            z2 = a*x
            yp1 = z1 + b
            yp2 = z2 + b
            w1 = yp1 - y
            w2 = yp2 - y
            loss = w1 * w2
            loss.backward()
            a = ad.Numtor(a.value - eta * a.grad)
            b = ad.Numtor(b.value - eta * b.grad)
    print(loss.value, a.value, b.value)
