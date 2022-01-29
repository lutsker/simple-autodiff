import autodiff as ad
import numpy as np


if __name__ == '__main__':
    print('Linear regression using SGD')
    N = 1000
    D = 100.0
    alpha = 2.45
    beta = -7.1
    xx = np.arange(1000) / float(N) * D
    yy = alpha * xx + beta + np.random.normal(loc=0, scale=0.5, size=N)

    # Model
    eta = 0.000001
    epochs = 5
    a = ad.Numtor(0.1)
    b = ad.Numtor(0.2)
    for _ in range(epochs):
        for sample_index in np.arange(N):
            x = ad.Numtor(xx[sample_index])
            y = ad.Numtor(yy[sample_index])
            yp1 = a*x + b
            yp2 = a*x + b
            loss = (yp1-y) * (yp2-y)
            loss.backward()
            a = ad.Numtor(a.value - eta * a.grad)
            b = ad.Numtor(b.value - eta * b.grad) 
        print(loss.value, a.value, b.value)
