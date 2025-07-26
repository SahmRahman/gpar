import torch
import gpytorch
from matplotlib import pyplot as plt

# Set manual seed for reproducibility
torch.manual_seed(0)

# --- Synthetic Data (2 Tasks, Each with 2 Inputs) ---
# Task 0: sin function with some noise
x0 = torch.linspace(0, 1, 30).unsqueeze(1)
x0 = torch.cat([x0, x0 ** 2], dim=1)  # 2 features
y0 = torch.sin(2 * torch.pi * x0[:, 0]) + 0.1 * torch.randn(x0.size(0))
task_idx0 = torch.zeros(x0.size(0), dtype=torch.long)

# Task 1: cos function with different noise
x1 = torch.linspace(0, 1, 30).unsqueeze(1)
x1 = torch.cat([x1, torch.sqrt(x1 + 1e-5)], dim=1)
y1 = torch.cos(2 * torch.pi * x1[:, 0]) + 0.1 * torch.randn(x1.size(0))
task_idx1 = torch.ones(x1.size(0), dtype=torch.long)

# Combine tasks
train_x = torch.cat([x0, x1], dim=0)
train_y = torch.cat([y0, y1], dim=0)
task_indices = torch.cat([task_idx0, task_idx1], dim=0)


# --- GP Model Definition ---
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_y, likelihood):
        super().__init__(train_inputs, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.data_covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_inputs[0].shape[1])
        )
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=2, rank=1)

    def forward(self, *inputs):  # Accepts unpacked args
        x, task_idx = inputs
        mean = self.mean_module(x)
        cov_data = self.data_covar_module(x)
        cov_task = self.task_covar_module(task_idx)
        cov = cov_data.mul(cov_task)
        return gpytorch.distributions.MultivariateNormal(mean, cov)


# --- Likelihood and Model ---
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = MultitaskGPModel((train_x, task_indices), train_y, likelihood)

# --- Training ---
model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 100
for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x, task_indices)  # <-- use unpacked inputs
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()

# --- Evaluation ---
model.eval()
likelihood.eval()

with torch.no_grad():
    preds = likelihood(model(train_x, task_indices))
    mean = preds.mean
    lower, upper = preds.confidence_region()

# --- Plotting ---
plt.figure(figsize=(10, 5))
for task_id in [0, 1]:
    idx = task_indices == task_id
    plt.plot(train_x[idx][:, 0], train_y[idx], 'k*', label=f'Train Task {task_id}')
    plt.plot(train_x[idx][:, 0], mean[idx], label=f'Pred Task {task_id}')
    plt.fill_between(train_x[idx][:, 0].numpy(),
                     lower[idx].numpy(),
                     upper[idx].numpy(), alpha=0.3)

plt.legend()
plt.title("Multitask GP with Task-Specific Inputs using IndexKernel")
plt.xlabel("x[0] (shared input feature)")
plt.ylabel("y")
plt.show()
