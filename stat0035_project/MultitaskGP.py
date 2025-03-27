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

for turbines in all_combinations():

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
                gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]), num_tasks=len(turbines), rank=1
            )
            self.covar_module.data_covar_module.lengthscale = torch.tensor(
                [([3.0] + [4.0] * (len(input_col_names) - 1)) * len(turbines)])

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=len(turbines))
    model = MultitaskGPModel(train_x, train_y, likelihood)


    # this is for running the notebook in our testing framework
    # smoke_test = ('CI' in os.environ)
    # training_iterations = 500

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    print(f"Training for turbines {turbines}")
    loss_delta = 99999
    prev_loss = loss_delta
    i = 0
    while loss_delta > 1:
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss_delta = prev_loss - loss
        prev_loss = loss
        loss.backward()
        # print('Iter %d - Loss: %.3f' % (i+1, loss.item()))
        optimizer.step()
        i += 1

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

    mean = mean.detach().numpy()
    lower = lower.detach().numpy()
    upper = upper.detach().numpy()
    test_y = test_y.numpy()

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
                'RMSE': np.float32(RMSE[j]),
                'MAE': np.float32(MAE[j]),
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
            file_path="/Users/sahmrahman/Desktop/GitHub/stat0035_project/Complete n=1000 run on Wind Speed, Direction and Temperature (while loop stop).pkl",
            new_row=new_row)

        # selected_test_x = test_x[:, j * len(turbines)].numpy()

        # gr.plot_graph(x=selected_test_x,
        #               y_list=[test_y[:, j].detach().numpy(),
        #                       mean[:, j],
        #                       upper[:, j],
        #                       lower[:, j]],
        #               model_history_index=-1,
        #               intervals=True,
        #               x_label="Wind Speed",
        #               y_label="Power",
        #               title=f'Turbine {turbines[j]}')

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
