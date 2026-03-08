import sys

from libraries import plt, np, os, datetime, ph, pd
from GPARModel import WindFarmGPAR

model_metadata_path = WindFarmGPAR.turbine_model_metadata_filepath


def contains_illegal_chars(value, name):
    illegal_characters = ['\\', '/', ':', '*', '?', '"', '<', '>', '|', '\0']
    if any(char in value for char in illegal_characters):
        raise ValueError(f"{name} contains illegal characters: {illegal_characters}")



def plot_graph(x, y_list,
               model_history_index=None,
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
               plot_within=True,
               x_date=True,
               legend_loc='upper right',
               fig_size=(12, 6)):

    illegal_characters = ['\\', '/', ':', '*', '?', '"', '<', '>', '|', '\0']

    # validate labels
    if x_label:
        contains_illegal_chars(x_label, "x_label")
    if y_label:
        contains_illegal_chars(y_label, "y_label")

    plt.figure(figsize=fig_size)

    if not intervals:

        for i, y in enumerate(y_list):

            color = colors[i] if colors and i < len(colors) else None
            label = labels[i] if labels and i < len(labels) else None

            if hollow:
                plt.scatter(x, y, label=label,
                            edgecolors='black',
                            facecolors='none',
                            marker='o')
            else:
                plt.scatter(x, y, label=label,
                            color=color,
                            marker='o')

    else:

        observations = np.array(y_list[0])
        uppers = np.array(y_list[-1])
        lowers = np.array(y_list[-2])
        x = np.array(x)

        # sort all x-dependent arrays
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]

        observations = observations[sorted_indices]
        uppers = uppers[sorted_indices]
        lowers = lowers[sorted_indices]

        # check which observations within CI
        inside_CI = (uppers > observations) & (observations > lowers)

        # Plot observations in/out CI
        if not hollow and plot_within:
            plt.scatter(x_sorted[inside_CI],
                        observations[inside_CI],
                        label="Observations inside CI",
                        c='black',
                        marker='o',
                        s=15)

        elif plot_within:
            plt.scatter(x_sorted[inside_CI],
                        observations[inside_CI],
                        label="Observations inside CI",
                        edgecolors='black',
                        facecolors='none',
                        marker='o',
                        s=15)

        plt.scatter(x_sorted[~inside_CI],
                    observations[~inside_CI],
                    label="Observations outside CI",
                    c='red',
                    marker='o',
                    s=60)

        # plot intermediate curves
        for i, y in enumerate(y_list):

            if 0 < i < len(y_list) - 2:

                y = np.array(y)[sorted_indices]

                color = colors[i] if colors and i < len(colors) else None
                label = labels[i] if labels and i < len(labels) else None

                if not hollow:
                    plt.scatter(x_sorted, y,
                                label=label,
                                color=color,
                                marker='o')
                else:
                    plt.scatter(x_sorted, y,
                                label=label,
                                edgecolors=color,
                                facecolors='none',
                                marker='o')

        # CI bars
        y_sorted = 0.5 * (uppers + lowers)

        yerr_lower = y_sorted - lowers
        yerr_upper = uppers - y_sorted

        plt.errorbar(
            x_sorted,
            y_sorted,
            yerr=[yerr_lower, yerr_upper],
            fmt='none',
            color='black',
            ecolor='lightblue',
            elinewidth=1,
            capsize=3,
            label='95% Confidence Interval'
        )

        # determine tick range
        if x_limits:
            start, end = x_limits
        else:
            start, end = x_sorted.min(), x_sorted.max()

        # generate ticks with step size 2
        tick_positions = list(np.arange(start, end, 2))

        # ensure the starting point is included
        if not tick_positions or tick_positions[0] != start:
            tick_positions.insert(0, start)

        # ensure the final bound is included
        if tick_positions[-1] != end:
            tick_positions.append(end)

        # labels
        if x_date:
            tick_labels = [pd.to_datetime(ts).strftime('%Y-%m-%d') for ts in tick_positions]
        else:
            tick_labels = [round(num, 2) for num in tick_positions]

        plt.xticks(tick_positions, tick_labels)

    if x_label:
        plt.xlabel(x_label)

    if y_label:
        plt.ylabel(y_label)

    if title:
        plt.title(title)
    else:
        plt.title(f"{x_label} vs {y_label} - Modelling History Index {model_history_index}")

    if x_limits:
        plt.xlim(x_limits)

    if y_limits:
        plt.ylim(y_limits)

    # legend handling
    if labels:
        handles, legend_labels = plt.gca().get_legend_handles_labels()

        if calibration > 0:
            handles.append(plt.Line2D([0], [0], linestyle="none"))
            legend_labels.append(f"Calibration {round(calibration * 100, 2)}%")

        plt.legend(handles, legend_labels, loc=legend_loc)

    plt.grid(False)

    if save_path:

        filename = title if title else \
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M')} - {x_label} vs {y_label} - Modelling History Index {model_history_index}"

        full_path = os.path.join(save_path, filename + '.png')

        plt.savefig(full_path)

        print(f"Figure saved at: {full_path}")

    else:
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


def plot_mtgp_metadata(indices, history_path="/Users/sahmrahman/Desktop/GitHub2/publication/MTGP Modelling History.pkl", save_path=''):

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
        mtgp_history = ph.read_pickle_as_dataframe("/Users/sahmrahman/Desktop/GitHub2/publication/Complete Runs/MTGP/Complete n=1000 run on Wind Speed, Direction and Temperature.pkl")
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
               fig_size=(18, 12),
               y_label="Power (kWh)"
               )

    return

def plot_wind_speed_forecast(test_data, gpar_history_indices, turbine, gpar_permutation, mtgp_combination=None, save_path=None, hollow=True, plot_within=True, legend_loc='upper center'):
    turbine_perm = [f'Turbine {i} Power' for i in gpar_permutation]
    test_data = test_data[test_data['turbine'] == turbine]

    gpar_history = ph.get_model_history().loc[gpar_history_indices]
    gpar_history = gpar_history[gpar_history['Output Columns'].apply(lambda x: len(x) == len(turbine_perm))]
    gpar_history = gpar_history[gpar_history['Output Columns'].apply(lambda x: x == turbine_perm)]

    if mtgp_combination:
        mtgp_history = ph.read_pickle_as_dataframe("/Users/sahmrahman/Desktop/GitHub2/publication/Complete Runs/MTGP/Complete n=1000 run on Wind Speed, Direction and Temperature.pkl")
        mtgp_history = mtgp_history[mtgp_history['Turbine Combination'].apply(lambda x: len(x) == len(mtgp_combination))]
        mtgp_history = mtgp_history[mtgp_history['Turbine Combination'].apply(lambda x: x == mtgp_combination)]
        mtgp_history = mtgp_history[mtgp_history['Turbine'] == turbine]

        test_data['MTGP Means'] = mtgp_history['Means'].iloc[0]
        test_data['MTGP Uppers'] = mtgp_history['Uppers'].iloc[0]
        test_data['MTGP Lowers'] = mtgp_history['Lowers'].iloc[0]

    test_data['GPAR Means'] = gpar_history['Means'].iloc[0][f'Turbine {turbine} Power']
    test_data['GPAR Uppers'] = gpar_history['Uppers'].iloc[0][f'Turbine {turbine} Power']
    test_data['GPAR Lowers'] = gpar_history['Lowers'].iloc[0][f'Turbine {turbine} Power']



    sorted_test_data = test_data.sort_values(by='Wind.speed.me', ascending=True)

    plot_graph(x=sorted_test_data['Wind.speed.me'],
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
               plot_within=plot_within,
               legend_loc=legend_loc,
               fig_size=(18, 12),
               x_date=False,
               y_label="Power (kWh)"
               )

    return


perm = [3,4,5,1,2,6]
output_sum = pd.read_csv("/Users/sahmrahman/Desktop/GitHub2/publication/Complete Runs/GPAR/Best Calibration/1k/Output Data - Whole Farm.csv")
test_data = ph.read_pickle_as_dataframe("/Users/sahmrahman/Desktop/GitHub2/publication/Test Sample.pkl")
plot_graph(x=test_data[test_data['turbine']==1]['Wind.speed.me'],
           y_list=[output_sum['Power.me'],
                   output_sum['Lowers'],
                   output_sum['Uppers'],
                   ],
           intervals=True,
           x_label="Wind Speed (mps)",
           y_label="Wind Farm Power (kwh)",
           title="Wind Farm Power Prediction on Wind Speed (read from Turbine 1) - 1k Training Data",
           save_path="/Users/sahmrahman/Desktop/GitHub2/publication/Complete Runs/GPAR/Best Calibration/1k/",
           plot_within=False,
           x_date=False,
           x_limits=(0,16),
           fig_size=(12,6))
