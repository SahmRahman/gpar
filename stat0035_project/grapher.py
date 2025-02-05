from libraries import plt, np, os, datetime, ph, pd

model_metadata_path = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/Turbine Model Metadata.pkl'


def contains_illegal_chars(value, name):
    illegal_characters = ['\\', '/', ':', '*', '?', '"', '<', '>', '|', '\0']
    if any(char in value for char in illegal_characters):
        raise ValueError(f"{name} contains illegal characters: {illegal_characters}")


def plot_graph(x, y_list, model_history_index,
               intervals=False,
               calibration=0,
               labels=None,
               colors=None,
               x_label=None,
               y_label=None,
               title=None,
               x_limits=None,
               y_limits=None,
               save_path=None):
    """
    Plots a graph using the given x values and multiple y datasets with optional customization.

    Parameters:
    - x: List or array-like, x-axis values
    - y_list: List of List or array-like, y-axis values for each dataset
    - model_history_index: int, index in Modelling History.pkl where this data was pulled
    - intervals: boolean, optional, will plot last two lists in y_list as an interval
    - calibration: float (between 0 and 1), optional, proportion of test data captured by intervals
    - labels: List of str, optional, labels for each dataset (excluding confidence interval)
    - colors: List of str, optional, colors for each dataset (excluding confidence interval)
    - x_label: str, optional, label for the x-axis
    - y_label: str, optional, label for the y-axis
    - title: str, optional, title of the graph
    - x_limits: tuple, optional, (min, max) limits for the x-axis
    - y_limits: tuple, optional, (min, max) limits for the y-axis
    - save_path: str, optional, directory to save the figure
    """

    illegal_characters = ['\\', '/', ':', '*', '?', '"', '<', '>', '|', '\0']

    # Validate labels
    if x_label:
        contains_illegal_chars(x_label, "x_label")
    if y_label:
        contains_illegal_chars(y_label, "y_label")

    plt.figure(figsize=(12, 6))

    # Plot each dataset

    if not intervals:  # no confidence intervals, just plot all points
        for i, y in enumerate(y_list):
            color = colors[i] if colors and i < len(colors) else None
            label = labels[i] if labels and i < len(labels) else None
            plt.scatter(x, y, label=label, color=color, marker='o')

    else:  # confidence intervals, plot last two lists in y_list as intervals
        observations = y_list[0]
        uppers = y_list[-1]
        lowers = y_list[-2]

        # --------- Plot observations and colour if outside CI ---------
        inside_CI = np.array(
            [True if uppers[i] > observations[i] > lowers[i] else False for i in range(len(observations))])

        x = np.array(x)
        observations = np.array(observations)

        plt.scatter(x[inside_CI], observations[inside_CI], label="Observations inside CI", c='black', marker='o', s=20)
        plt.scatter(x[~inside_CI], observations[~inside_CI], label="Observations outside CI", c='red', marker='o', s=50)

        # --------- take care of anything between observations and bounds ---------

        for i, y in enumerate(y_list):
            if 0 < i < len(y_list) - 2:
                color = colors[i] if colors and i < len(colors) else None
                label = labels[i] if labels and i < len(labels) else None
                plt.scatter(x, y, label=label, color=color, marker='o')

        # --------- plot confidence interval ---------

        uppers = np.array(uppers)
        lowers = np.array(lowers)

        sorted_indices = np.argsort(x)  # Get indices to sort x in ascending order
        x_sorted = x[sorted_indices]  # Sort x-values
        uppers_sorted = uppers[sorted_indices]  # Sort uppers values according to sorted x
        lowers_sorted = lowers[sorted_indices]  # Sort lowers values according to sorted x

        # Fill between uppers and lowers
        plt.fill_between(x_sorted, uppers_sorted, lowers_sorted, color='lightblue', alpha=0.5,
                         label='95% Confidence Interval')

    # Set labels and title if provided
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    if title:
        plt.title(title)
    else:
        plt.title(f"{x_label} vs {y_label} - Modelling History Index {model_history_index}")

    # Set axis limits if provided
    if x_limits:
        plt.xlim(x_limits)
    if y_limits:
        plt.ylim(y_limits)

    if labels:
        if calibration > 0:
            # Get the existing legend handles and labels
            handles, labels = plt.gca().get_legend_handles_labels()

            # Add a dummy handle for the custom text
            handles.append(plt.Line2D([0], [0], linestyle="none"))  # Invisible handle
            labels.append(f"Calibration {round(calibration * 100, 2)}%")  # Custom text to add

            # Update the legend with the new entry
            plt.legend(handles, labels, loc="lower right")

        else:  # no calibration, just make the legend as normal
            plt.legend(loc='lower right')  # Add legend in the top-left corner if labels are provided

    plt.grid(True)  # Add a grid for better readability

    # Save the figure if a directory is provided
    if save_path:
        filename = ''

        if title:
            filename = title
        else:
            filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M')} - {x_label} vs {y_label} - Modelling History Index {model_history_index}"
        # replace "_" with " " if a title is given
        # otherwise just do "X vs. Y - Modelling History Index z - 1999-01-01_00-59.png"

        full_path = os.path.join(save_path, filename + '.png')
        plt.savefig(full_path)
        print(f"Figure saved at: {full_path}")

    else:
        # only show if not saving
        plt.show()


def plot_model_metadata(indices=[], save_path=''):
    df_model_metadata = ph.read_pickle_as_dataframe(model_metadata_path)
    selected_metadata = df_model_metadata.iloc[indices]

    turbines = []
    indices_by_permutation_size = {1: [],
                                   2: [],
                                   3: [],
                                   4: [],
                                   5: [],
                                   6: []}

    for index, row in selected_metadata.iterrows():

        permutation = row['Turbine Permutation']

        indices_by_permutation_size[len(permutation)].append(index)

        for turbine in permutation:
            if turbine not in turbines:
                turbines.append(turbine)

    for turbine in turbines:
        plt.figure(figsize=(8, 6))
        plt.xlabel('Turbine Permutation Size')
        plt.ylabel('Error')
        plt.title(f'Model Metadata for Turbine {turbine}')

        markers = ['o', 's', '^', 'D', 'p', '*']
        i = 0

        for perm_size, indices in indices_by_permutation_size.values():

            if indices:
                # non-empty index list
                # i.e. we actually have data for this perm_size and turbine

                current_data = selected_metadata.iloc[indices]

                MSE = current_data['MSE']
                MAE = current_data['MAE']

                plt.scatter(x=perm_size,
                            y=MSE,
                            label=f'MSE; {perm_size} Turbines',
                            color='black',
                            marker=markers[i])

                plt.scatter(x=perm_size,
                            y=MAE,
                            label=f'MAE; {perm_size} Turbines',
                            color='blue',
                            marker=markers[i])

                i += 1

        if save_path:
            filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M')} Model Metadata for " \
                       f"Turbine {turbine} at Indices {indices}"

            full_path = os.path.join(save_path, filename + '.png')
            plt.savefig(full_path)
            print(f"Figure saved at: {full_path}")
        else:
            plt.show()
