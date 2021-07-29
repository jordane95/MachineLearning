import numpy as np

a = np.array([1, 2, 3])
b = np.array([2, 3, 4])

np.savez("test", a=a, b=b)

np.savez("test", a=b, b=a)

npz = np.load('test.npz')

a_ = npz['a']
b_ = npz['b']


print(a_, b_)
