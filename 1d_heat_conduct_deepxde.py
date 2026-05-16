import deepxde as dde
import numpy as np
from config import L, k, Q, T0, TL

# PDE: -k * d2T/dx2 = Q
def pde(x):
    T = x[:, 0:1]
    dT_dx = dde.grad.jacobian(T, x, i=0, j=0)
    d2T_dx2 = dde.grad.hessian(T, x, i=0, j=0)
    return -k*d2T_dx2 - Q

# Boundary conditions
def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)

def boundary_right(x, on_boundary):
    return on_boundary and np.isclose(x[0], L)

# Define the geometry and time domain
geom = dde.geometry.Interval(0, L)
# Define the PDE problem
data = dde.data.PDE(
    geom, pde, 
    [
        dde.DirichletBC(geom, lambda x: T0, boundary_left),
        dde.DirichletBC(geom, lambda x: TL, boundary_right)
    ], 
num_domain=100, num_boundary=2)

# Define the neural network
net = dde.nn.FNN([1] + [50] * 3 + [1], "tanh", "Glorot normal")

# Define the model
model = dde.Model(data, net)
# Compile and train the model
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(epochs=5000)







