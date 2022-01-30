# Simple autodiff

execute tests with `pytest`.

The autodiff operates on real numbers, represented by class `Numtor`.
A few operations are implemented. 

Usage:
``` 
import autodiff as ad 

a = ad.Numtor(1)
b = ad.Numtor(3)
r = a * b
r.backward()

print(a.grad, b.grad)
```
This snippet will calculate and evaluate the derivatives $\frac{dr}{da} = b = 3$ and $\frac{dr}{db} = a = 1$.
