
import numpy as np

a = np.random.rand(2, 512, 512)

print(a.shape)
out = np.average(a, axis=0)
out = np.expand_dims(out, axis=0)

print((out==a).all())
print(out)
print(out.shape)