import math
import torch
import gpytorch
import os
from libraries import ph, pd, np, gr
from matplotlib import pyplot as plt

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

# train_x = torch.linspace(0, 1, 100)
#
# train_y = torch.stack([
#     torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
#     torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
# ], -1)

train_sample = ph.read_pickle_as_dataframe(
    "/Users/sahmrahman/Desktop/GitHub/stat0035_project/Training Sample.pkl")
test_sample = ph.read_pickle_as_dataframe(
    "/Users/sahmrahman/Desktop/GitHub/stat0035_project/Test Sample.pkl")

input_col_names = ['Wind.speed.me', "Wind.dir.sin.me", 'Wind.dir.cos.me', 'Nacelle.ambient.temp.me']

turbines = (1,2,3,4)

train_x = torch.from_numpy(pd.concat(
    [train_sample[train_sample['turbine'] == i][input_col_names].reset_index(drop=True) for i in turbines],
    axis=1
).to_numpy()).to(torch.float32)
# gather the input columns into a dataframe per turbine,
# then append them together column-wise and convert into one big numpy ndarray

train_y = torch.from_numpy(pd.concat(
    [train_sample[train_sample['turbine'] == i]['Power.me'].reset_index(drop=True) for i in turbines],
    axis=1
).to_numpy()).to(torch.float32)

test_x = torch.from_numpy(pd.concat(
    [test_sample[test_sample['turbine'] == i][input_col_names].reset_index(drop=True) for i in turbines],
    axis=1
).to_numpy()).to(torch.float32)

test_y = torch.from_numpy(pd.concat(
    [test_sample[test_sample['turbine'] == i]['Power.me'].reset_index(drop=True) for i in turbines],
    axis=1
).to_numpy()).to(torch.float32)


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=len(turbines)
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=len(turbines), rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=len(turbines))
model = MultitaskGPModel(train_x, train_y, likelihood)

# this is for running the notebook in our testing framework
smoke_test = ('CI' in os.environ)
training_iterations = 2 if smoke_test else 50

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
    optimizer.step()

# Set into eval mode
model.eval()
likelihood.eval()


# Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # test_x = torch.linspace(0, 1, 51)
    predictions = likelihood(model(test_x))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()

# This contains predictions for both tasks, flattened out
# The first half of the predictions is for the first task
# The second half is for the second task

for j in [0, 1, 2, 3]:

    selected_test_x = test_x[:, j * 4].numpy()

    gr.plot_graph(x=selected_test_x,
                  y_list=[test_y[:, j].detach().numpy(),
                          mean[:, j],
                          upper[:, j],
                          lower[:, j]],
                  model_history_index=-1,
                  intervals=True,
                  x_label="Wind Speed",
                  y_label="Power",
                  title=f'Turbine {turbines[j]}')

    # # Initialize plots
    # fig, ((y1_ax, y2_ax), (y3_ax, y4_ax)) = plt.subplots(2, 2, figsize=(16, 10))
    #
    # for ax in [y1_ax, y2_ax, y3_ax, y4_ax]:
    #
    #     # selected_train_x = train_x[:, j*4+j]
    #     selected_test_x = test_x[:, j*4+j].numpy()
    #
    #     # Plot test data as black stars
    #     ax.plot(selected_test_x, test_y[:, j].detach().numpy(), 'k*')
    #
    #     sorted_indices = np.argsort(selected_test_x)  # Get indices to sort x in ascending order
    #     x_sorted = selected_test_x[sorted_indices]  # Sort x-values
    #     means_sorted = mean[sorted_indices, j]  # Sort mean values according to sorted x
    #     uppers_sorted = upper[sorted_indices, j]  # Sort upper values according to sorted x
    #     lowers_sorted = lower[sorted_indices, j]  # Sort lower values according to sorted x
    #
    #     # Predictive mean as blue line
    #     ax.plot(x_sorted, means_sorted, 'b')
    #
    #     # Shade in confidence
    #     ax.fill_between(x_sorted, lowers_sorted, uppers_sorted, alpha=0.5)
    #
    #     # ax.set_ylim([-3, 3])
    #     ax.legend([input_col_names[j], 'Mean', 'Confidence'])
    #     ax.set_title(f'{input_col_names[j]} for Turbine {turbines[j]}')
    #
    # plt.show()
