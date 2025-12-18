import torch
import torch.nn as nn
import numpy as np

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define neural network
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.net(x)

# RHS function f(x)
def f(x):
    return torch.sin(np.pi * x)

# Model
model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training points
N = 100
x = torch.linspace(0, 1, N).view(-1, 1).to(device)
x.requires_grad = True

# Boundary points
x0 = torch.tensor([[0.0]], requires_grad=True).to(device)
x1 = torch.tensor([[1.0]], requires_grad=True).to(device)

# Training loop
epochs = 5000
for epoch in range(epochs):

    optimizer.zero_grad()

    # Prediction
    u = model(x)

    # First derivative
    du_dx = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    # Second derivative
    d2u_dx2 = torch.autograd.grad(
        du_dx, x, grad_outputs=torch.ones_like(du_dx),
        create_graph=True
    )[0]

    # PDE residual
    pde_loss = torch.mean((-d2u_dx2 + u - f(x))**2)

    # Neumann BCs
    du_dx_0 = torch.autograd.grad(model(x0), x0,
                                  grad_outputs=torch.ones_like(x0),
                                  create_graph=True)[0]

    du_dx_1 = torch.autograd.grad(model(x1), x1,
                                  grad_outputs=torch.ones_like(x1),
                                  create_graph=True)[0]

    bc_loss = du_dx_0.pow(2).mean() + du_dx_1.pow(2).mean()

    # Total loss
    loss = pde_loss + bc_loss
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.6f}")

print("Training complete!")
