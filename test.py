import numpy as np

# print(np.array([np.random.uniform(-1, 1) for i in range(8)]))


# w = np.array([-0.10046567, 0.13037817, 0.39261494]).reshape(3, 1)
# print(w)


# inputS = np.array([1, 1, 1])
# print(inputS)


# print(inputS.dot(w))

# x = np.array([1, 2, 3, 4]).reshape(4, 1)
# y = np.array([1, 2, 3, 4]).reshape(4, 1)
# v = np.array([np.random.uniform(-10, 10)
#               for i in range(len(x))]).reshape(len(x), 1)
# print(v)
# print(v + (x-y) * 2)

import time

time.perf_counter()
for i in range(10000000):
    r = 2**100
x = time.perf_counter()
print(x)
