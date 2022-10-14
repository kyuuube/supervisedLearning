import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Prints numpy arrays nicer
np.set_printoptions(precision=2, suppress=True, linewidth=100)

def target_function(x):
    return x * 0.5 - 4


num_samples = 30
# Randomly sampled values in [-10, 10]
xs = np.random.uniform(low=-10, high=10, size=num_samples)
# Intended target value plus random noise
ys = target_function(xs) + np.random.normal(loc=0, scale=1, size=num_samples)

data = np.array(list(zip(xs, ys)))
print('data:')
print(data)

plt.figure(dpi=150)
plt.title('Data')
plt.xlabel('x')
plt.ylabel('y')
plt.plot([-12, 12], [target_function(-12), target_function(12)],
         color='#458588', label='target_function')
plt.scatter(xs, ys, color='#458588', label='data')
plt.legend()
plt.show(block=True)
plt.close()