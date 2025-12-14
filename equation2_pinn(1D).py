import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Define f(x)
# -----------------------------
def f(x):
    return np.ones_like(x)   # f(x) = 1

# -----------------------------
# 2. Define PDE: -u'' = f
# -----------------------------
def f(x):
    return 1.0

def pde(x, u):
    u_xx = dde.grad.hessian(u, x)
    return -u_xx - f(x)

# -----------------------------
# 3. Geometry
# -----------------------------
geom = dde.geometry.Interval(0, 1)

# -----------------------------
# 4. Neumann Boundary Conditions
# u'(0)=0, u'(1)=0
# -----------------------------
bc_left = dde.NeumannBC(
    geom,
    lambda x: 0,
    lambda x, on_boundary: on_boundary and np.isclose(x[0], 0)
)

bc_right = dde.NeumannBC(
    geom,
    lambda x: 0,
    lambda x, on_boundary: on_boundary and np.isclose(x[0], 1)
)

# -----------------------------
# 5. Data
# -----------------------------
data = dde.data.PDE(
    geom,
    pde,
    [bc_left, bc_right],
    num_domain=100,
    num_boundary=20
)

# -----------------------------
# 6. Neural Network
# -----------------------------
net = dde.nn.FNN([1, 40, 40, 1], "tanh", "Glorot normal")
model = dde.Model(data, net)

# -----------------------------
# 7. Train
# -----------------------------
model.compile("adam", lr=0.001)
model.train(epochs=5000)

# -----------------------------
# 8. Plot Solution
# -----------------------------
x = np.linspace(0, 1, 200)[:, None]
u_pred = model.predict(x)

plt.plot(x, u_pred)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("PINN Solution of Equation 2 (Neumann BC)")
plt.show()

