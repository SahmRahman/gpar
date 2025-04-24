from libraries import plt, np, os, datetime, ph, pd
from GPARModel import WindFarmGPAR

model_metadata_path = WindFarmGPAR.turbine_model_metadata_filepath


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
               save_path=None,
               hollow=True,
               legend_loc='upper right',
               fig_size = (12,6)):
    """
    Plots a graph using the given x values and multiple y datasets with optional customization.

    Parameters:
    - x: List or array-like, x-axis values
    - y_list: List of List or array-like, y-axis values for each dataset
    - model_history_index: int, index in Modelling History X.pkl where this data was pulled
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
    - hollow: boolean, optional, hollow points
    - fig_size: tuple, optional, figure size of plot
    """

    illegal_characters = ['\\', '/', ':', '*', '?', '"', '<', '>', '|', '\0']

    # Validate labels
    if x_label:
        contains_illegal_chars(x_label, "x_label")
    if y_label:
        contains_illegal_chars(y_label, "y_label")

    plt.figure(figsize=fig_size)

    # Plot each dataset

    if not intervals:  # no confidence intervals, just plot all points
        for i, y in enumerate(y_list):
            color = colors[i] if colors and i < len(colors) else None
            label = labels[i] if labels and i < len(labels) else None
            if hollow:
                plt.scatter(x, y, label=label, edgecolors='black', facecolors='none', marker='o')
            else:
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

        if not hollow:
            plt.scatter(x[inside_CI], observations[inside_CI], label="Observations inside CI", c='black', marker='o', s=35)
        else:
            plt.scatter(x[inside_CI], observations[inside_CI], label="Observations inside CI", edgecolors='black', facecolors='none', marker='o', s=35)

        plt.scatter(x[~inside_CI], observations[~inside_CI], label="Observations outside CI", c='red', marker='o', s=60)
        # this should always be coloured in, it's meant to draw attention

        # --------- take care of anything between observations and bounds ---------

        for i, y in enumerate(y_list):
            if 0 < i < len(y_list) - 2:
                color = colors[i] if colors and i < len(colors) else None
                label = labels[i] if labels and i < len(labels) else None
                if not hollow:
                    plt.scatter(x, y, label=label, color=color, marker='o')
                else:
                    plt.scatter(x, y, label=label, edgecolors=color, facecolors='none', marker='o')

        # --------- plot confidence interval ---------

        uppers = np.array(uppers)
        lowers = np.array(lowers)

        sorted_indices = np.argsort(x)  # Get indices to sort x in ascending order
        x_sorted = x[sorted_indices]  # Sort x-values
        uppers_sorted = uppers[sorted_indices]  # Sort uppers values according to sorted x
        lowers_sorted = lowers[sorted_indices]  # Sort lowers values according to sorted x

        # Compute central value and asymmetric error
        y_sorted = 0.5 * (uppers_sorted + lowers_sorted)
        yerr_lower = y_sorted - lowers_sorted
        yerr_upper = uppers_sorted - y_sorted

        # Plot vertical bars with horizontal caps
        plt.errorbar(
            x_sorted, y_sorted,
            yerr=[yerr_lower, yerr_upper],
            fmt='none', color='black', ecolor='lightblue', elinewidth=1, capsize=3,
            label='95% Confidence Interval'
        )

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
            plt.legend(handles, labels, loc=legend_loc)

        else:  # no calibration, just make the legend as normal
            plt.legend(loc=legend_loc)  # Add legend in the top-left corner if labels are provided

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

    plt.close()


def plot_model_metadata(indices=[], save_path=''):
    df_model_metadata = ph.read_pickle_as_dataframe(model_metadata_path)
    selected_metadata = df_model_metadata.loc[indices]

    turbines = []
    entries_by_permutation_size = {}

    for index, row in selected_metadata.iterrows():

        permutation = row['Turbine Permutation']

        if str(len(permutation)) in entries_by_permutation_size.keys():
            entries_by_permutation_size[str(len(permutation))].append(row)
        else:
            entries_by_permutation_size[str(len(permutation))] = [row]

        for turbine in permutation:
            if turbine not in turbines:
                turbines.append(turbine)

    for i in entries_by_permutation_size.keys():
        entries_by_permutation_size[i] = pd.DataFrame(entries_by_permutation_size[i])
        # convert each list of DataFrame rows to one full DataFrame

    for metadata_val, y_lims in zip(['MSE', 'MAE', 'Calibration'],
                                    [(30, 200), (30, 150), (.7, 1)]):
        for turbine in turbines:

            plt.figure(figsize=(8, 6))
            plt.xlabel('Turbine Permutation Size')
            if metadata_val == 'MSE':
                plt.ylabel('R' + metadata_val)
                title = f'Model R{metadata_val} for Turbine {turbine} by Permutation Size'
            else:
                plt.ylabel(metadata_val)
                title = f'Model {metadata_val} for Turbine {turbine} by Permutation Size'
            plt.title(title)
            plt.xlim((0, 7))
            plt.ylim(y_lims)
            plt.grid(True)

            metadata = [entries_by_permutation_size[i][entries_by_permutation_size[i]['Turbine'] == turbine][metadata_val]
                        for i in entries_by_permutation_size.keys()]

            plt.boxplot(x=metadata, positions=[int(i) for i in entries_by_permutation_size.keys()])

            if metadata_val == 'Calibration':
                plt.axhline(y=0.95, color='r', linestyle='-', linewidth=1.5, label='Confidence Interval %')
                plt.legend(loc='lower right')

            if save_path:
                filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M')} " + title

                full_path = os.path.join(save_path, filename + '.png')
                plt.savefig(full_path)
                print(f"Figure saved at: {full_path}")
            else:
                plt.show()

            plt.close()


def plot_mtgp_metadata(indices, history_path="/Users/sahmrahman/Desktop/GitHub/stat0035_project/MTGP Modelling History.pkl", save_path=''):

    metadata = ph.read_pickle_as_dataframe(history_path)
    selected_metadata = metadata[metadata.index.isin(indices)]

    turbines = []
    entries_by_combination_size = {}

    for index, row in selected_metadata.iterrows():

        combination = row['Turbine Combination']

        if str(len(combination)) in entries_by_combination_size.keys():
            entries_by_combination_size[str(len(combination))].append(row)
        else:
            entries_by_combination_size[str(len(combination))] = [row]

        for turbine in combination:
            if turbine not in turbines:
                turbines.append(turbine)

    for i in entries_by_combination_size.keys():
        entries_by_combination_size[i] = pd.DataFrame(entries_by_combination_size[i])
        # convert each list of DataFrame rows to one full DataFrame

    for metadata_val, y_lims in zip(['RMSE', 'MAE'],  # , 'Calibration'],
                                    [(30, 200), (30, 150)]):  # , (.7, 1)]):
        for turbine in turbines:

            plt.figure(figsize=(8, 6))
            plt.xlabel('Turbine Combination Size')
            title = f'Model {metadata_val} for Turbine {turbine} by Combination Size'
            plt.title(title)
            plt.xlim((0, 7))
            plt.ylim(y_lims)
            plt.grid(True)

            metadata = [
                entries_by_combination_size[i][entries_by_combination_size[i]['Turbine'] == turbine][metadata_val]
                for i in entries_by_combination_size.keys()
            ]

            plt.boxplot(x=metadata, positions=[int(i) for i in entries_by_combination_size.keys()])

            # if metadata_val == 'Calibration':
            #     plt.axhline(y=0.95, color='r', linestyle='-', linewidth=1.5, label='Confidence Interval %')
            #     plt.legend(loc='lower right')

            if save_path:
                filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M')} " + title

                full_path = os.path.join(save_path, filename + '.png')
                plt.savefig(full_path)
                print(f"Figure saved at: {full_path}")
            else:
                plt.show()

            plt.close()


def print_model_metadata(indices=[]):
    df_model_metadata = ph.read_pickle_as_dataframe(model_metadata_path)
    selected_metadata = df_model_metadata.loc[indices]

    columns = ['Turbine',
               'Permutation Size',
               'Mean MSE', 'MSE Std. Dev.', 'Max MSE', 'Max MSE Perm.', 'Min MSE', 'Min MSE Perm.',
               'Mean MAE', 'MAE Std. Dev.', 'Max MAE', 'Max MAE Perm.', 'Min MAE', 'Min MAE Perm.',
               'Mean Calib.', 'Calib. Std. Dev.', 'Max Calib.', 'Max Calib. Perm.', 'Min Calib.', 'Min Calib. Perm.']

    # df = pd.DataFrame(columns=columns)

    # Calculate the column width based on the longest element in list1
    column_width = max(len(str(col)) for col in columns) + 6

    print()
    for col in columns:
        print(f"{col:<{column_width}}", end="")
    print()
    print('-'*column_width*len(columns))  # add a line of "----" for formatting

    for turbine in range(1, 7):
        for perm_size in range(1, 7):

            row = [turbine, perm_size]
            current_data = selected_metadata[selected_metadata['Turbine'] == turbine]
            current_data = current_data[current_data['Turbine Count'] == perm_size]

            for metadata_metric in ('MSE', 'MAE', 'Calibration'):

                if not current_data.empty:

                    metadata_values = current_data[metadata_metric].values.tolist()
                    mean = round(np.mean(metadata_values), 3)
                    std = round(np.std(metadata_values), 3)
                    max_index = current_data[metadata_metric].idxmax()
                    min_index = current_data[metadata_metric].idxmin()
                    max_val = round(current_data.loc[max_index][metadata_metric], 3)
                    min_val = round(current_data.loc[min_index][metadata_metric], 3)
                    max_perm = current_data.loc[max_index]['Turbine Permutation']
                    min_perm = current_data.loc[min_index]['Turbine Permutation']

                    # -------------- Figured I'd keep this if I want to save this as a DataFrame --------------
                    # data_to_append[f'Mean {metadata_metric}'] = mean
                    # data_to_append[f'{metadata_metric} Std. Dev.'] = std
                    # data_to_append[f'Best Permutation by {metadata_metric}'] = current_data[current_data[metadata_metric] == max]['Turbine Permutation']
                    # data_to_append[f'Worst Permutation by {metadata_metric}'] = current_data[current_data[metadata_metric] == min]['Turbine Permutation']

                    # actually I shouldn't need this, I can just zip the columns and row together

                    row += [mean, std, max_val, max_perm, min_val, min_perm]


            for element in row:
                print(f"{str(element):<{column_width}}", end="")
            print()  # Move to the next row after each iteration

        print()  # just to have a blank line to break up the turbines


def plot_forecast_comparison(test_data, gpar_history_indices, turbine, gpar_permutation, mtgp_combination=None, save_path=None, hollow=True, legend_loc='upper center'):
    turbine_perm = [f'Turbine {i} Power' for i in gpar_permutation]
    test_data = test_data[test_data['turbine'] == turbine]

    gpar_history = ph.get_model_history().loc[gpar_history_indices]
    gpar_history = gpar_history[gpar_history['Output Columns'].apply(lambda x: len(x) == len(turbine_perm))]
    gpar_history = gpar_history[gpar_history['Output Columns'].apply(lambda x: x == turbine_perm)]

    if mtgp_combination:
        mtgp_history = ph.read_pickle_as_dataframe("/Users/sahmrahman/Desktop/GitHub/stat0035_project/Complete Runs/MTGP/Complete n=1000 run on Wind Speed, Direction and Temperature.pkl")
        mtgp_history = mtgp_history[mtgp_history['Turbine Combination'].apply(lambda x: len(x) == len(mtgp_combination))]
        mtgp_history = mtgp_history[mtgp_history['Turbine Combination'].apply(lambda x: x == mtgp_combination)]
        mtgp_history = mtgp_history[mtgp_history['Turbine'] == turbine]

        test_data['MTGP Means'] = mtgp_history['Means'].iloc[0]
        test_data['MTGP Uppers'] = mtgp_history['Uppers'].iloc[0]
        test_data['MTGP Lowers'] = mtgp_history['Lowers'].iloc[0]

    test_data['GPAR Means'] = gpar_history['Means'].iloc[0][f'Turbine {turbine} Power']
    test_data['GPAR Uppers'] = gpar_history['Uppers'].iloc[0][f'Turbine {turbine} Power']
    test_data['GPAR Lowers'] = gpar_history['Lowers'].iloc[0][f'Turbine {turbine} Power']



    sorted_test_data = test_data.sort_values(by='Date.time', ascending=True)

    plot_graph(x=sorted_test_data['Date.time'],
               y_list=[sorted_test_data['Power.me'].values,
                       # sorted_test_data['MTGP Means'].values,
                       sorted_test_data['GPAR Lowers'].values,
                       sorted_test_data['GPAR Uppers'].values],
                       # sorted_test_data['MTGP Lowers'],
                       # sorted_test_data['MTGP Uppers']],
               labels=['Observations', 'GPAR Upper', 'GPAR Lower'],
               colors=['black', 'blue', 'red'],
               title=f'Prediction for Turbine {turbine} with Permutation {gpar_permutation}',
               model_history_index=-1,
               intervals=True,
               save_path=save_path,
               hollow=hollow,
               legend_loc=legend_loc,
               fig_size=(10, 6),
               y_label="Power (kWh)"
               )