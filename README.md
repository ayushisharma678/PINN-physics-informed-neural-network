# PINN-physics-informed-neural-network
Using PINN(Physics Informed Neural Network) to solve Aeronautical and Thermal differential equations

## Equation 2: Poisson Equation with Neumann Boundary Condition

This project solves the Poisson equation using a Physics-Informed Neural Network (PINN).

### Governing Equation
-Δu = f in Ω  
∂u/∂n = 0 on ∂Ω  

### Tools Used
- Python
- DeepXDE
- PyTorch

### Result
The PINN successfully approximates the solution while satisfying the Neumann boundary conditions.

![Solution](equation2_solution.png)

