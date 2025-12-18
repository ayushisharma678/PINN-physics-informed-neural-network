import torch
import torch.nn as nn

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
