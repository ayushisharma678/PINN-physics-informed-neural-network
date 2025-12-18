import deepxde as dde
import numpy as np

# 1D domain
geom = dde.geometry.Interval(0, 1)

# Source term
def f(x):
    return 1.0

# PDE: -u'' + u = 1
def pde(x, u):
    u_xx = dde.grad.hessian(u, x)
    return -u_xx + u - f(x)

# Dirichlet boundary: u = 0
def boundary(x, on_boundary):
    return on_boundary

bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)

# Data
data = dde.data.PDE(
    geom,
    pde,
    bc,
    num_domain=100,
    num_boundary=20
)

# Neural network
net = dde.nn.FNN([1] + [50]*3 + [1], "tanh", "Glorot normal")

model = dde.Model(data, net)

model.compile("adam", lr=0.001)
model.train(iterations=5000)

# Plot solution
losshistory, train_state = model.train(iterations=5000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)


