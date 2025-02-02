import libraries as libs
from libraries import np


class WindFarmGPAR:
    __modelling_history_filepath = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/Modelling History.pkl'
    __models_filepath = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/Models.pkl'
    __turbine_model_metadata_filepath = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/Turbine Model Metadata.pkl'

    def __init__(self, model_params, existing, model_index):
        """
        initialiser for a model

        :param existing: whether model already exists, boolean
        :param model_index: index of model if it exists, int
        :param model_params: parameter name:value for the GPARRegressor model, dictionary
        :param train_size: sample size of training data, int
        :param test_size: sample size of test data, int
        """

        self.model = WindFarmGPAR.create_model(existing, model_params, model_index)

        if model_index > -1:
            self.model_index = model_index
        else:
            self.model_index = len(libs.pkl.read_pickle_as_dataframe(WindFarmGPAR.__models_filepath)) - 1
            # need to have - 1 even if it's a new model b/c after the create_model() call,
            # it would have been added to the Models.pkl file

    @staticmethod
    def create_model(existing, model_params, model_index):
        """
        create GPARRegressor model, either from scratch or rebuild

        :param existing: model exists already, bool
        :param model_params: model parameters, dictionary
        :param model_index: index in model pickle/dataframe, int
        :return: model, GPARRegressor instance
        """

        if existing and model_index > -1:
            # model exists, pull from pickle file @ given index

            models_df = libs.pkl.read_pickle_as_dataframe(WindFarmGPAR.__models_filepath)
            model_params = models_df.iloc[model_index]

            model_params = model_params[model_params.notna()]

            # only returns parameters that won't be their respective default values

        else:
            # need to make from scratch using model_params

            # add new model to our models file
            libs.pkl.append_to_pickle(WindFarmGPAR.__models_filepath, model_params)

        model = libs.GPARRegressor(**model_params)
        # ** is python's way of unpacking a dictionary as parameters

        # ================== GPARRegressor parameters ==================

        """
        replace (bool, optional):    
            Replace observations with predictive means.
            Helps the model deal with noisy data points.
            Defaults to `False`.


        impute (bool, optional):    
            Impute data with predictive means to make the
            data set closed downwards.
            Helps the model deal with missing data.
            Defaults to `True`.


        scale (tensor, optional):    
            Initial value(s) for the length scale(s) over the
            inputs.
            Defaults to `1.0`.


        scale_tie (bool, optional):    
            Tie the length scale(s) over the inputs.
            Defaults to `False`.


        per (bool, optional):    
            Use a locally periodic kernel over the inputs.
            Defaults to `False`.


        per_period (tensor, optional):    
            Initial value(s) for the period(s) of the
            locally periodic kernel.
            Defaults to `1.0`.


        per_scale (tensor, optional):    
            Initial value(s) for the length scale(s) of the
            locally periodic kernel.
            Defaults to `1.0`.


        per_decay (tensor, optional):    
            Initial value(s) for the length scale(s) of the
            local change of the locally periodic kernel.
            Defaults to `10.0`.


        input_linear (bool, optional):    
            Use a linear kernel over the inputs.
            Defaults to `False`.


        input_linear_scale (tensor, optional):    
            Initial value(s) for the length
            scale(s) of the linear kernel over the inputs.
            Defaults to `100.0`.


        linear (bool, optional):    
            Use linear dependencies between outputs.
            Defaults to `True`.


        linear_scale (tensor, optional):    
            Initial value(s) for the length scale(s) of
            the linear dependencies.
            Defaults to `100.0`.


        nonlinear (bool, optional):    
            Use nonlinear dependencies between outputs.
            Defaults to `True`.


        nonlinear_scale (tensor, optional):    
            Initial value(s) for the length scale(s)
            over the outputs.
            Defaults to `0.1`.


        rq (bool, optional):    
            Use rational quadratic (RQ) kernels instead of
            exponentiated quadratic (EQ) kernels.
            Defaults to `False`.


        markov (int, optional):    
            Markov order of conditionals.
            Set to `None` to have a fully connected structure.
            Defaults to `None`.


        noise (tensor, optional):    
            Initial value(s) for the observation noise(s).
            Defaults to `0.01`.


        x_ind (tensor, optional):    
            Locations of inducing points.
            Set to `None` if inducing points should not be used.
            Defaults to `None`.


        normalise_y (bool, optional):    
            Normalise outputs.
            Defaults to `True`.


        transform_y (tuple, optional):    
            Tuple containing a transform and its
            inverse, which should be applied to the data before fitting.
            Defaults to the identity transform.
"""

        return model

    @staticmethod
    def sample_data(train_df, test_df, train_size=100, test_size=10):
        train_indices = libs.np.random.choice(train_df[['index']].values.flatten(), train_size)
        test_indices = libs.np.random.choice(test_df[['index']].values.flatten(), test_size)
        # get random sample without replacement from 'index' values in train/test dataframes
        # of the given sizes

        train_sample = train_df[train_df['index'].isin(train_indices)]
        test_sample = test_df[test_df['index'].isin(test_indices)]
        # select those rows

        return {"train": train_sample,
                "test": test_sample}

    @staticmethod
    def sample_split_data(train_df, test_df, split_columns, train_size=100, test_size=10):

        train_sample = []
        test_sample = []

        for col in split_columns:
            col_values = train_df[col].unique().tolist()
            for value in col_values:
                selected_train_df = train_df[train_df[col] == value]
                selected_test_df = test_df[test_df[col] == value]

                train_indices = libs.np.random.choice(selected_train_df[['index']].values.flatten(), train_size)
                test_indices = libs.np.random.choice(selected_test_df[['index']].values.flatten(), test_size)

                train_sample.append(selected_train_df[selected_train_df['index'].isin(train_indices)])
                test_sample.append(selected_test_df[selected_test_df['index'].isin(test_indices)])
                # append a new dataframe where column matches current value, and then take the sample
        #
        # train_sample = libs.pd.concat(train_sample, ignore_index=True)
        # test_sample = libs.pd.concat(test_sample, ignore_index=True)

        return {'train': train_sample,
                'test': test_sample}

    @staticmethod
    def specify_data(df, columns):
        """
        reshapes selected inputs to be agreeable with GPARs methods

        :param df: input values, pandas DataFrame
        :param columns: input columns, list of strings
        :return: ndarray of selected columns from the dataframe, numpy ndarray
        """
        return df[columns].values

    @staticmethod
    def log_results(results_df, input_cols, output_cols, training_indices, test_indices, model_index,
                    turbine_permutation):
        """
        log the results of a model run

        :param results_df: outputs, pandas DataFrame
        :param input_cols: input columns, list of strings
        :param output_cols: output columns, list of strings
        :param training_indices: indices of the original training data dataframe, list of ints
        :param test_indices: indices of the original test data dataframe, list of ints
        :param model_index: model index, int
        :param turbine_permutation: order of turbines in model, list of ints
        :return: NONE, simply saves results to pickle file and prints the filename
        """

        # Get the current timestamp
        timestamp = libs.datetime.now().strftime("%Y-%m-%d_%H-%M")

        results_df['Input Columns'] = [input_cols]
        results_df['Output Columns'] = [output_cols]
        results_df['Training Data Indices'] = [training_indices]
        results_df['Test Data Indices'] = [test_indices]
        # need to wrap the above in lists so their element in the dataframe is itself a list
        # (rather than multiple entries of the same column)

        results_df['Model Index'] = model_index
        results_df['Timestamp'] = timestamp

        # Save the DataFrame as a Pickle file

        libs.pkl.append_to_pickle(file_path=WindFarmGPAR.__modelling_history_filepath,
                                  new_row=results_df)

        for col in output_cols:

            absolute_error = np.array(results_df['Error'].iloc[0][col]['Absolute Error'],
                                      ndmin=2)
            interval_half_width = (np.array(results_df['Uppers'].iloc[0][col], ndmin=2) - np.array(results_df['Lowers'].iloc[0][col], ndmin=2)) / 2

            inside = absolute_error < interval_half_width
            calibration = np.sum(inside, axis=1)[0] / inside.shape[1]
            # sum along axis 1 (column) (for some reason, the ndmin forces it to be p x n instead of n x p)
            # take the first element (np.sum() returns a ndarray of one value)
            # then divide by n = shape[1]

            MSE = np.sqrt(np.mean(results_df['Error'].iloc[0][col]['Squared Error']))
            MAE = np.mean(results_df['Error'].iloc[0][col]['Absolute Error'])

            df_modelling_history = libs.pkl.read_pickle_as_dataframe(file_path=WindFarmGPAR.__modelling_history_filepath)
            model_metadata = {
                'Turbine Count': len(turbine_permutation),
                'Turbine Permutation': turbine_permutation,
                'Modelling History Index': len(df_modelling_history) - 1,
                'Model Index': model_index,
                'Calibration': calibration,
                'MSE': MSE,
                'MAE': MAE
            }

            libs.pkl.append_to_pickle(file_path=WindFarmGPAR.__turbine_model_metadata_filepath,
                                      new_row=model_metadata)

    def train_model(self, train_x, train_y,
                    test_x, test_y,
                    train_indices, test_indices,
                    input_columns, output_columns,
                    turbine_permutation=[]):
        """
        fit model to train data, draw samples from resulting posterior and log the results
        :param output_columns: names of outputs, list of strings
        :param input_columns: names of inputs, list of strings
        :param train_x: training input data, ndarray
        :param train_y: training output data, ndarray
        :param test_x: test input data, ndarray
        :param test_y: test output data, ndarray
        :param train_indices: indices in the training pickle file used for this model, list of ints
        :param test_indices: indices in the test pickle file used for this model, list of ints
        :param turbine_permutation: order of turbines in model, list of ints
        :return: NONE
        """

        # for cols in [input_columns, output_columns]:
        #     if 'index' in cols:
        #         cols.remove('index')

        # sample_dict = self.sample_data()
        # train_sample = sample_dict['train']
        # test_sample = sample_dict['test']
        # establish training and testing data for GPARRegressor model
        # want to preserve these to be pandas DataFrames

        # adjust shape/data type of data sample to be ndarrays for GPAR
        # training_input = WindFarmGPAR.specify_data(train_sample, input_columns)
        # training_output = WindFarmGPAR.specify_data(train_sample, output_columns)
        # test_input = WindFarmGPAR.specify_data(test_sample, input_columns)
        # test_output = WindFarmGPAR.specify_data(test_sample, output_columns)

        # train model
        self.model.fit(train_x, train_y)

        # collect metadata
        means, lowers, uppers = self.model.predict(test_x,
                                                   num_samples=50,
                                                   credible_bounds=True)

        error = {"SE": (means - test_y) ** 2,
                 "AE": np.absolute(means - test_y)}
        # dictionary of errors
        # first pair is ndarray of squared errors per prediction
        # second pair is absolute error

        # ================ organise and log results ================

        metadata = {
            # uses dictionary comprehensions
            # ------- {key:  value                for key,value in iterable}
            "Means": [{name: means[:, i].tolist() for i, name in enumerate(output_columns)}],
            "Lowers": [{name: lowers[:, i].tolist() for i, name in enumerate(output_columns)}],
            "Uppers": [{name: uppers[:, i].tolist() for i, name in enumerate(output_columns)}],
            "Error": [{name: {"Squared Error": error['SE'][:, i].tolist(),
                              "Absolute Error": error['AE'][:, i].tolist()} for i, name in enumerate(output_columns)}]
        }
        # dictionary with list values (of one length),
        # each list consists of one dictionary where the output column
        #   is matched to its respective statistic list
        # e.g. metadata['Means'][0]['Power.me'] = list of sample means for Power.me
        # had to do additional list wrapping to not screw with dataframe element length

        # ------ "Error"'s value is a bit ugly, but it's just storing the error dictionary (as defined ~25 lines above)
        # ------ per output column

        #
        # train_indices = train_sample[['index']].values.flatten()
        # test_indices = test_sample[['index']].values.flatten()

        metadata = libs.pd.DataFrame(metadata)
        # now, metadata['Means'].iloc[0]['Power.me'] = list of sample means for Power.me
        # ---- notice difference between [0] and .iloc[0] because we reference a row in a dataframe now
        # ---- rather than just an element in a list

        WindFarmGPAR.log_results(
            results_df=metadata,
            input_cols=input_columns,
            output_cols=output_columns,
            training_indices=train_indices,
            test_indices=test_indices,
            model_index=self.model_index,
            turbine_permutation=turbine_permutation
        )
