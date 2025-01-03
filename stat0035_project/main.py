# -*- coding: utf-8 -*-
"""STAT0035 Project.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1EIyv_OmfhwVD_5KTqAZvXIKv1-a_vwvZ
"""

import libraries as libs


class WindFarmGPAR:
    __modelling_history_filepath = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/Modelling History.pkl'
    __models_filepath = '/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/GitHub/stat0035_project/Models.pkl'

    def __init__(self, train_data_path, test_data_path, model_params, existing, model_index):
        """
        initialiser for a model

        :param train_data_path: filepath, string
        :param test_data_path: filepath, string
        :param existing: whether model already exists, boolean
        :param model_index: index of model if it exists, int
        :param model_params: parameter name:value for the GPARRegressor model, dictionary
        """
        self.train_data = libs.pd.read_pickle(train_data_path)
        self.test_data = libs.pd.read_pickle(test_data_path)

        # need to have some control over missing filepath

        self.model = WindFarmGPAR.create_model(existing, model_params, model_index)

        if model_index > -1:
            self.model_index = model_index
        else:
            self.model_index = len(libs.pkl.read_pickle_as_dataframe(WindFarmGPAR.__models_filepath)) - 1
            # need to have - 1 even if it's a new model b/c by now it would have been added to the Models.pkl file

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

        model = libs.gpar.GPARRegressor(**model_params)
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
            
        """

        return model

    def sample_data(self):
        train_sample = self.train_data.iloc[:100]
        test_sample = self.test_data.iloc[:100]

        return {"train": train_sample,
                "test": test_sample}

    @staticmethod
    def specify_data(df, columns):
        """
        reshapes selected inputs to be agreeable with GPARs methods

        :param df: input values, pandas DataFrame
        :param columns: input columns, list of strings
        :return: 2D ndarray of selected columns from the dataframe, numpy ndarray
        """
        return df[columns].values

    @staticmethod
    def log_results(results_df, input_cols, output_cols, training_indices, test_indices, model_index):
        """
        log the results of a model run

        :param results_df: outputs, pandas DataFrame
        :param input_cols: input columns, list of strings
        :param output_cols: output columns, list of strings
        :param training_indices: indices of the original training data dataframe, list of ints
        :param test_indices: indices of the original test data dataframe, list of ints
        :param model_index: model index, int
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

    def train_model(self, input_columns, output_columns):
        """
        1) sample data of requested input and output columns
        2) create the GPARRegressor model
        3) train the model
        4) store the results

        :param input_columns: input columns, list of strings
        :param output_columns: output columns, list of strings
        :return: NONE, trains model and logs the results
        """

        for cols in [input_columns, output_columns]:
            if 'index' in cols:
                cols.remove('index')

        sample_dict = self.sample_data()
        train_sample = sample_dict['train']
        test_sample = sample_dict['test']
        # establish training and testing data for GPARRegressor model
        # want to preserve these to be pandas DataFrames

        # adjust shape/data type of data sample to be ndarrays for GPAR
        training_input = WindFarmGPAR.specify_data(train_sample, input_columns)
        training_output = WindFarmGPAR.specify_data(train_sample, output_columns)
        test_input = WindFarmGPAR.specify_data(test_sample, input_columns)
        test_output = WindFarmGPAR.specify_data(test_sample, output_columns)

        # train model
        self.model.fit(training_input, training_output)

        # collect metadata
        means, lowers, uppers = self.model.predict(test_input,
                                                   num_samples=5,
                                                   credible_bounds=True)

        error = means - test_output

        metadata = {
            "Means": [means.flatten()],
            "Lowers": [lowers.flatten()],
            "Uppers": [uppers.flatten()],
            "Error": [error.flatten()]
        }

        # need to cast as whole list so lengths don't mess up

        # organise and log results
        train_indices = train_sample[['index']].values.flatten()
        test_indices = test_sample[['index']].values.flatten()

        metadata = libs.pd.DataFrame(metadata)

        WindFarmGPAR.log_results(
            results_df=metadata,
            input_cols=input_columns,
            output_cols=output_columns,
            training_indices=train_indices,
            test_indices=test_indices,
            model_index=self.model_index
        )


model_obj = WindFarmGPAR(
    train_data_path="/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/Wind farm final year project _ SR_DL_PD/train.pkl",
    test_data_path="/Users/sahmrahman/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3 UCL/STAT0035/Wind farm final year project _ SR_DL_PD/test.pkl",
    model_index=0,
    existing=True,
    model_params={})

input_cols = [
    'Wind.dir.std',
    'Wind.speed.me',
    'Wind.speed.sd',
    'Wind.speed.min',
    'Wind.speed.max',
    'Front.bearing.temp.me',
    'Front.bearing.temp.sd',
    'Front.bearing.temp.min',
    'Front.bearing.temp.max',
    'Rear.bearing.temp.me',
    'Rear.bearing.temp.sd',
    'Rear.bearing.temp.min',
    'Rear.bearing.temp.max',
    'Stator1.temp.me',
    'Nacelle.ambient.temp.me',
    'Nacelle.temp.me',
    'Transformer.temp.me',
    'Gear.oil.inlet.temp.me',
    'Gear.oil.temp.me',
    'Top.box.temp.me',
    'Hub.temp.me',
    'Conv.Amb.temp.me',
    'Rotor.bearing.temp.me',
    'Transformer.cell.temp.me',
    'Motor.axis1.temp.me',
    'Motor.axis2.temp.me',
    'CPU.temp.me',
    'Blade.ang.pitch.pos.A.me',
    'Blade.ang.pitch.pos.B.me',
    'Blade.ang.pitch.pos.C.me',
    'Gear.oil.inlet.press.me',
    'Gear.oil.pump.press.me',
    'Drive.train.acceleration.me',
    'Tower.Acceleration.x',
    'Tower.Acceleration.y',
    'Wind.dir.sin.me',
    'Wind.dir.cos.me',
    'Wind.dir.sin.min',
    'Wind.dir.cos.min',
    'Wind.dir.sin.max',
    'Wind.dir.cos.max'
]

model_obj.train_model(input_columns=input_cols,
                      output_columns=['Power.me'])

# scale=0.1,            Initial length scale for the inputs.
# linear=True,          Use linear dependencies between outputs.
# linear_scale=10.0,    Length scale for linear dependencies between outputs.
# nonlinear=True,       Also use nonlinear dependencies between outputs.
# nonlinear_scale=0.1,  Length scale for nonlinear dependencies (post-normalisation, if enabled).
# noise=0.1,            Variance of the observation noise.
# impute=True,          Impute missing data to ensure data is closed downwards.
# replace=False,        Do not replace data points with posterior mean of previous layer (retains noise).
# normalise_y=False     Work with raw outputs, without normalising them.
