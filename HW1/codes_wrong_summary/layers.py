import numpy as np


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''

        self._saved_tensor = tensor


class Selu(Layer):
    def __init__(self, name):
        super(Selu, self).__init__(name)
        self.params = {
            "lambda": 1.0507,
            "alpha": 1.67326
        }

    def forward(self, input):
        # TODO START
        """
        u > 0, lamb * u -> lamb
        u <= 0, lamb * alpha * (e^u - 1) -> lamb * alpha * e^u
        """
        self._saved_for_backward(input)
        ret = np.where(input > 0,
                       self.params["lambda"] * input,
                       self.params["lambda"] * self.params["alpha"] * (np.exp(input) - 1))
        return ret.copy()
        # TODO END

    def backward(self, grad_output):
        # TODO START
        saved_input = self._saved_tensor
        ret = (grad_output *
               self.params["lambda"] *
               np.where(saved_input > 0,
                        1,
                        self.params["alpha"] * np.exp(saved_input)))
        return ret.copy()
        # TODO END


class HardSwish(Layer):
    def __init__(self, name):
        super(HardSwish, self).__init__(name)

    def forward(self, input):
        # TODO START
        """
        u <= -3 , 0 -> 0
        u >= 3 u -> 1
        otherwise (u * (u + 3)) / 6 -> (2u + 3) / 6
        """
        self._saved_for_backward(input)
        conditions = [input <= -3, input >= 3, np.logical_and(input > -3, input < 3)]
        choices = [0, input, input * (input + 3) / 6]
        ret = np.select(conditions, choices)
        return ret.copy()
        # TODO END

    def backward(self, grad_output):
        # TODO START
        saved_input = self._saved_tensor
        conditions = [saved_input <= -3, saved_input >= 3, np.logical_and(saved_input > -3, saved_input < 3)]
        choices = [0, 1, (2 * saved_input + 3) / 6]
        ret = grad_output * np.select(conditions, choices)
        return ret.copy()
        # TODO END


class Tanh(Layer):
    def __init__(self, name):
        super(Tanh, self).__init__(name)

    def forward(self, input):
        # TODO START
        self._saved_for_backward(input)
        """
        tanh = e - e^- / e + e^-
        -> 1 - tanh^2
        """

        ret = np.tanh(input)
        return ret.copy()

        # TODO END

    def backward(self, grad_output):
        # TODO START
        saved_input = self._saved_tensor

        ret = grad_output * (1 - np.tanh(saved_input) ** 2)
        return ret.copy()
        # TODO END


class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

    def forward(self, input):
        # TODO START
        self._saved_for_backward(input)
        ret = input @ self.W + self.b
        return ret.copy()

        # TODO END

    def backward(self, grad_output):
        # TODO START
        """
        X * W + b -> W = X^trans * grad
        """
        saved_input = self._saved_tensor
        self.grad_W = saved_input.T @ grad_output
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True)

        ret = grad_output @ self.W.T
        return ret.copy()
        # TODO END

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b
