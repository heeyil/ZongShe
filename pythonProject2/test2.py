from tensor import *

x = Tensor([1, 2, 3])
y = Tensor([3, 2, 1])
z = x * y
t = sum(z)
m = exp(t)
print(f"m={m.data}")

x = Tensor([1., 2., 3.], requires_grad=True)
y = log(x) + x
z = sum(y)
z.backward()
print(x.grad)



