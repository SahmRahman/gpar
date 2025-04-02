import math
import torch
import gpytorch
import os
from libraries import ph, pd, np
import grapher as gr
from matplotlib import pyplot as plt
from itertools import combinations

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
train_sample_indices = train_sample.index
test_sample = ph.read_pickle_as_dataframe(
    "/Users/sahmrahman/Desktop/GitHub/stat0035_project/Test Sample.pkl")
test_sample_indices = test_sample.index


def all_combinations(numbers=range(1, 7)):
    """Generate all combinations of numbers for all lengths."""
    result = []
    for length in range(1, len(numbers) + 1):
        result.extend(combinations(numbers, length))
    return result


input_col_names  = ['Wind.speed.me', "Wind.dir.sin.me", 'Wind.dir.cos.me', 'Nacelle.ambient.temp.me']

train_n = 990
test_n = 96

for turbines in all_combinations():

    train_x = torch.from_numpy(pd.concat(
        [train_sample[train_sample['turbine'] == i][input_col_names].reset_index(drop=True) for i in turbines],
        axis=0
    ).to_numpy()).to(torch.float32)
    # gather the input columns into a dataframe per turbine,
    # then append them together column-wise and convert into one big numpy ndarray

    task_indices = torch.cat([torch.ones(train_n) * i for i in range(len(turbines))])
    # create long torch arrays of the corresponding index (e.g. [0,0,0,...,1,1,1,...,2,2,2,...])
    # and concatenate them by row

    train_y = torch.from_numpy(pd.concat(
        [train_sample[train_sample['turbine'] == i]['Power.me'].reset_index(drop=True) for i in turbines],
        axis=0
    ).to_numpy()).to(torch.float32)

    test_x = torch.from_numpy(pd.concat(
        [test_sample[test_sample['turbine'] == i][input_col_names].reset_index(drop=True) for i in turbines],
        axis=0
    ).to_numpy()).to(torch.float32)

    test_y = torch.from_numpy(pd.concat(
        [test_sample[test_sample['turbine'] == i]['Power.me'].reset_index(drop=True) for i in turbines],
        axis=0
    ).to_numpy()).to(torch.float32)



    class MultitaskGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_inputs, train_y, likelihood):
            super().__init__(train_inputs, train_y, likelihood)

            self.mean_module = gpytorch.means.ConstantMean()

            self.data_covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=train_inputs[0].shape[1])
                # train_inputs[0] is train_x, train_inputs[1] are the task indices
            )

            self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=len(turbines), rank=1)


        def forward(self, *inputs):
            x, task_idxs = inputs
            mean = self.mean_module(x)
            cov_data = self.data_covar_module(x)
            cov_task = self.task_covar_module(task_idxs)
            cov = cov_data.mul(cov_task)
            return gpytorch.distributions.MultivariateNormal(mean, cov)

            # self.covar_module = gpytorch.kernels.MultitaskKernel(
            #     gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]), num_tasks=len(turbines), rank=1
            # )
            # self.covar_module.data_covar_module.lengthscale = torch.tensor(
            #     [([3.0] + [4.0] * (len(input_col_names) - 1)) * len(turbines)])

    # Define likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = MultitaskGPModel(train_inputs=(train_x, task_indices),
                             train_y=train_y,
                             likelihood=likelihood)

    # Training loop setup
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    print(f"Training for turbines {turbines}")
    loss_delta = 99999
    prev_loss = loss_delta * 2
    i = 0
    while abs(loss_delta/prev_loss) > .0001:  # if absolute change is less than .01%, stop!
        optimizer.zero_grad()
        output = model(train_x, task_indices)
        loss = -mll(output, train_y)
        loss_delta = prev_loss - loss
        prev_loss = loss
        loss.backward()
        print('Iter %d - Loss: %.3f' % (i + 1, loss.item()))
        optimizer.step()
        i += 1

    # Set into eval mode
    model.eval()
    likelihood.eval()

    # Make new task_indices with appropriate size for test data
    task_indices = torch.cat([torch.ones(test_n) * i for i in range(len(turbines))])

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # test_x = torch.linspace(0, 1, 51)
        predictions = likelihood(model(test_x, task_indices))
        mean = predictions.mean
        lower, upper = predictions.confidence_region()

    # This contains predictions for both tasks, flattened out
    # The first half of the predictions is for the first task
    # The second half is for the second task

    mean = mean.detach().numpy().reshape(mean.size(0) // test_n, test_n).T
    lower = lower.detach().numpy().reshape(lower.size(0) // test_n, test_n).T
    upper = upper.detach().numpy().reshape(upper.size(0) // test_n, test_n).T
    test_y = test_y.numpy().reshape(test_y.size(0) // test_n, test_n).T

    RMSE = np.sqrt(np.mean((mean - test_y) ** 2, axis=0))
    MAE = np.mean(np.abs(mean - test_y), axis=0)
    cal = np.mean(np.logical_and(upper > test_y, test_y > lower), axis=0)
    # print(f"Calibration: {cal}")

    for j in range(len(turbines)):
        new_row = {}

        if len(turbines) == 1:
            new_row = {
                'Turbine': turbines[j],
                'Turbine Combination': turbines,
                'Combination Size': len(turbines),
                'Means': mean.flatten(),
                'Uppers': upper.flatten(),
                'Lowers': lower.flatten(),
                'RMSE': np.float32(RMSE[0]),
                'MAE': np.float32(MAE[0]),
                'Input Columns': input_col_names,
                'Output Columns': [f'Turbine {turbines[j]} Power'],
                'Training Data Indices': train_sample_indices,
                'Test Data Indices': test_sample_indices
                # 'Calibration': cal
            }

        else:
            new_row = {
                'Turbine': turbines[j],
                'Turbine Combination': turbines,
                'Combination Size': len(turbines),
                'Means': mean[:, j],
                'Uppers': upper[:, j],
                'Lowers': lower[:, j],
                'RMSE': np.float32(RMSE[j]),
                'MAE': np.float32(MAE[j]),
                'Input Columns': input_col_names,
                'Output Columns': f'Turbine {turbines[j]} Power',
                'Training Data Indices': train_sample_indices,
                'Test Data Indices': test_sample_indices
                # 'Calibration': cal
            }

        ph.append_to_pickle(
            file_path="/Users/sahmrahman/Desktop/GitHub/stat0035_project/Complete n=1000 run on Wind Speed, Direction and Temperature (fixed hopefully).pkl",
            new_row=new_row
        )

