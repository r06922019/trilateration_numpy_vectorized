#!/usr/env/python3
import time
import numpy as np # np 1.17.2

# ref: www.101computing.net/cell-phone-trilateration-algorithm/
def trilateration(x1, y1, r1, x2, y2, r2, x3, y3, r3):
    A = 2*x2 - 2*x1
    B = 2*y2 - 2*y1
    C = r1**2 - r2**2 - x1**2 + x2**2 - y1**2 + y2**2
    D = 2*x3 - 2*x2
    E = 2*y3 - 2*y2
    F = r2**2 - r3**2 - x2**2 + x3**2 - y2**2 + y3**2
    x = (C*E - F*B) / (E*A - B*D)
    y = (C*D - A*F) / (B*D - A*E)
    return (x, y)


def generate_lists(n_points=100, n_list=3):
    rand_list = np.random.rand(n_list * n_points)
    return np.split(rand_list * 5, 3, axis=0)


def trilateration_vector(radar1, radar2, radar3):
    x1, y1, r1 = radar1
    r1 = np.square(r1)
    x2, y2, r2 = radar2
    r2 = np.square(r2)
    x3, y3, r3 = radar3
    r3 = np.square(r3)

    n_points = r1.size * r2.size * r3.size

    A = 2*x2 - 2*x1
    B = 2*y2 - 2*y1

    # C = r1**2 - r2**2
    r12 = np.tile(np.tile(r1, r2.size), r3.size)
    r22 = np.tile(np.repeat(r2, r1.size, axis=0), r3.size)
    C = r12 - r22
    C = C - x1**2 + x2**2 - y1**2 + y2**2

    D = 2*x3 - 2*x2
    E = 2*y3 - 2*y2

    # F = r2**2 - r3**2
    # F = np.repeat( np.repeat(np.expand_dims(r2, axis=1), r1.size, axis=1).reshape([-1, 1]), r3.size, axis=0) - np.repeat(np.expand_dims(r3, axis=1), r1.size*r2.size, axis=1).reshape([-1, 1])
    r32 = np.repeat(r3, r1.size*r2.size, axis=0)
    F = r22 - r32
    F = F - x2**2 + x3**2 - y2**2 + y3**2

    x = (C*E - F*B) / (E*A - B*D)
    y = (C*D - A*F) / (B*D - A*E)
    return np.concatenate([x.reshape([-1, 1]), y.reshape([-1, 1])], axis=1)


# generate 100 point list * 3
x1, x2, x3, y1, y2, y3 = (np.random.rand(6) - 0.5) * 5
r1_list, r2_list, r3_list = generate_lists()

# compute with general method
start_time = time.time()
point_list_general = [trilateration(x1, y1, r1, x2, y2, r2, x3, y3, r3) for r3 in r3_list for r2 in r2_list for r1 in r1_list]
print("%.2fs" % (time.time() - start_time))

# compute with vectorized method
radar1 = (x1, y1, r1_list)
radar2 = (x2, y2, r2_list)
radar3 = (x3, y3, r3_list)

start_time = time.time()
point_list_vector = trilateration_vector(radar1, radar2, radar3)
print("%.2fs" % (time.time() - start_time))

# compare error
print("mse:", np.mean(np.square(point_list_general - point_list_vector)))

