import pandas as pd


pd.set_option('display.max_columns', None)


class Model:

    def __init__(self, input_data, output_data):
        self.data = input_data
        self.output = output_data
        self.parameters = {
            'A': 1,
            'B': 2,
            'C': 3
        }
        self.results = pd.DataFrame(columns=['output', 'error', 'input', 'parameter', 'model'])

    # def __dict__(self):
    #     return {
    #         'name':'foo',
    #         'data': self.data,
    #         'output': self.output
    #     }


    def log_result(self, result, error, input_cols, parameter, object_dict):
        self.results.loc[len(self.results)] = [result, error, input_cols, parameter, object_dict]

    def train_model(self, data_choice, column_choice):

        result = self.data[data_choice]*self.parameters[column_choice]

        error = self.output[data_choice] - result

        obj_dict = self.__dict__
        print(obj_dict)

        self.log_result(result, error, data_choice, column_choice, obj_dict)


tester = Model([-1, 0, 5, 2, 9],
               [7, 8, 9, 10, 11])

tester.train_model(2, 'B')
tester.train_model(2, 'A')
tester.train_model(2, 'C')
tester.train_model(1, 'B')

print(tester.results)

