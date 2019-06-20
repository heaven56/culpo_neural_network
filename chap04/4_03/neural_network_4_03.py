import numpy as np
class neuralNetwork:

    def __init__(
        self,
        input_neurons,
        hidden_neurons,
        output_neurons,
        learning_rate
        ):
        '''
        ニューラルネットワークの初期化を行う
        '''
        #入力層、隠れ層、出力層のニューロン数をインスタンス変数に代入
        self.inneurons = input_neurons
        self.hneurons = hidden_neurons
        self.oneurons = output_neurons
        self.lr = learning_rate
        self.weight_initializer()

    def weight_initializer(self):
        '''
        隠れ層の重みとバイアスを初期化
        '''
        self.w1 = np.random.normal(
            0.0,
            pow(self.inneurons, -0.5),
            (self.hneurons,
             self.inneurons + 1)
            )

        self.w2 = np.random.normal(
            0.0,
            pow(self.inneurons, -0.5),
            (self.oneurons,
             self.hneurons + 1)
            )

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def softmax(self, x):
        c = np.max(x)
        exp_x = np.exp(x - c)
        sum_exp_x = np.sum(exp_x)
        y = exp_x / sum_exp_x
        return y

    def train(self, inputs_list, targets_list):
        inputs = np.array(
            np.append(inputs_list, [1]),
            ndmin = 2
        ).T

        hidden_inputs = np.dot(
            self.w1,
            inputs
            )

        hidden_outputs = self.sigmoid(hidden_inputs)
        hidden_outputs = np.append(
            hidden_outputs,
            [[1]],
            axis = 0
            )

        final_inputs = np.dot(
            self.w2,
            hidden_outputs
            )

        final_outputs = self.softmax(final_inputs)

        targets = np.array(
            targets_list,
            ndmin = 2
        ).T

        output_errors = final_outputs - targets

        delta_output = output_errors * (1 - final_outputs) * final_outputs

        hidden_errors = np.dot(
            self.w2.T,
            delta_output
            )

        self.w2 -= self.lr * np.dot(
            delta_output,
            hidden_outputs.T
            )

        hidden_errors_nobias = np.delete(
            hidden_errors,
            self.hneurons,
            axis = 0
            )

        hidden_outputs_nobias = np.delete(
            hidden_outputs,
            self.hneurons,
            axis = 0
            )

        self.w1 -= self.lr * np.dot(
            hidden_errors_nobias * (1.0 - hidden_errors_nobias) * hidden_outputs_nobias,
            inputs.T
            )

    def evaluate(self, inputs_list):
        inputs = np.array(
            np.append(inputs_list, [1]),
            ndmin = 2
            ).T

        hidden_inputs = np.dot(self.w1, inputs)

        hidden_outputs = self.sigmoid(hidden_inputs)

        final_inputs = np.dot(self.w2, np.append(hidden_outputs, [1]))

        final_outputs = self.softmax(final_inputs)

        return final_outputs