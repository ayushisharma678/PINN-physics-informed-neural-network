import numpy as np
import torch
import torch.nn as nn
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N = 100
x = np.linspace(0, 1, N)
h = x[1] - x[0]

# RHS
f = np.sin(np.pi * x)

# Matrix A
A = np.zeros((N, N))

for i in range(1, N-1):
    A[i, i-1] = -1 / h**2
    A[i, i]   = 2 / h**2 + 1
    A[i, i+1] = -1 / h**2

# Neumann BCs
A[0, 0] = 1
A[0, 1] = -1

A[-1, -1] = 1
A[-1, -2] = -1

f[0] = 0
f[-1] = 0

# Solve
u = np.linalg.solve(A, f)



x_train = torch.tensor(x, dtype=torch.float32).view(-1, 1)
u_train = torch.tensor(u, dtype=torch.float32).view(-1, 1)

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.net(x)

model = NN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(3000):
    optimizer.zero_grad()
    pred = model(x_train)
    loss = loss_fn(pred, u_train)
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.6f}")

print("NN training complete!")

# Evaluate trained model
x_test = torch.linspace(0, 1, 100).view(-1, 1).to(device)

with torch.no_grad():
    u_pred = model(x_test)

# Print some values
for i in range(0, 100, 20):
    print(f"x = {x_test[i].item():.2f}, u(x) = {u_pred[i].item():.6f}")

import matplotlib.pyplot as plt

plt.plot(x_test.cpu(), u_pred.cpu())
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("PINN solution of -u'' + u = f")
plt.show()
