import matplotlib.pyplot as plt
import numpy as np
from gpar.regression import GPARRegressor
from wbml.experiment import WorkingDirectory
import wbml.plot

if __name__ == "__main__":
    wd = WorkingDirectory("test_run", seed = 1)

    # Create toy data set.
    n = 200
    x = np.linspace(0, 1, n)
    noise = 0.1

    # Draw functions depending on each other in complicated ways.
    f1 = -np.cos(10 * np.pi * (x + 1)) / (2 * x + 1) - np.exp(x)
    f2 = np.cos(f1) ** 2 + np.sin(x)
    f3 = f2 * f1 ** 2 + 3 * x**5
    f = np.stack((f1, f2, f3), axis=0).T

    # Add noise and subsample.
    y = f + noise * np.random.randn(n, 3)
    x_obs, y_obs = x[::8], y[::8]

    # Fit and predict GPAR.
    model = GPARRegressor(
        scale=0.1,              # Initial length scale for the inputs.
        linear=True,            # Use linear dependencies between outputs.
        linear_scale=10.0,      # Length scale for linear dependencies between outputs.
        nonlinear=True,         # Also use nonlinear dependencies between outputs.
        nonlinear_scale=0.1,    # Length scale for nonlinear dependencies (post-normalisation, if enabled).
        noise=0.1,              # Variance of the observation noise.
        impute=True,            # Impute missing data to ensure data is closed downwards.
        replace=False,          # Do not replace data points with posterior mean of previous layer (may retain noise).
        normalise_y=False       # Work with raw outputs, without normalising them.
    )

    model.fit(x_obs, y_obs)
    means, lowers, uppers = model.predict(
        x, num_samples=100, credible_bounds=True, latent=True
    )

    # Fit and predict independent GPs: set `markov=0` in GPAR.
    igp = GPARRegressor(
        scale=0.1,
        linear=True,
        linear_scale=10.0,
        nonlinear=True,
        nonlinear_scale=0.1,
        noise=0.1,
        markov=0,
        normalise_y=False,
    )
    igp.fit(x_obs, y_obs)
    igp_means, igp_lowers, igp_uppers = igp.predict(
        x, num_samples=100, credible_bounds=True, latent=True
    )

    # Plot the result.
    plt.figure(figsize=(15, 3))

    for i in range(3):
        plt.subplot(1, 3, i + 1)

        # Plot observations.
        plt.scatter(x_obs, y_obs[:, i], label="Observations", style="train")
        plt.plot(x, f[:, i], label="Truth", style="test")

        # Plot GPAR.
        plt.plot(x, means[:, i], label="GPAR", style="pred")
        plt.fill_between(x, lowers[:, i], uppers[:, i], style="pred")

        # Plot independent GPs.
        plt.plot(x, igp_means[:, i], label="IGP", style="pred2")
        plt.fill_between(x, igp_lowers[:, i], igp_uppers[:, i], style="pred2")

        plt.xlabel("$t$")
        plt.ylabel(f"$y_{i + 1}$")
        wbml.plot.tweak(legend=i == 2)

    plt.tight_layout()
    plt.savefig(wd.file("synthetic.pdf"))
