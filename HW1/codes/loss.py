from __future__ import division
import numpy as np

# for debug
def print_line(inp):
    for i in range(inp.shape[0]):
        print(inp[i])


# for better code style
def softmax(input, axis=1):
    # 防爆 from https://blog.csdn.net/qq_39478403/article/details/116862070
    # max_predict = (batch, 10) -> (batch, 1)
    # get max value
    max_predict = np.max(input, axis=axis, keepdims=True)

    # (batch, 10) exp
    input_exp = np.exp(input - max_predict)

    # (batch, 1) sum
    input_exp_sum = np.sum(input_exp, axis=axis, keepdims=True)

    # (batch, 10 = [...h_k])
    return input_exp / input_exp_sum

class KLDivLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START

        batch = input.shape[0]

        input_softmax = softmax(input)
        input_softmax = np.clip(input_softmax, 1e-9, 1.0)  # 避免 input_softmax 为零

        # -inf 修正 否则会爆仓
        ln_t_sub_h = np.where(target == 0, 1e-9, np.log(target / input_softmax))

        kl_div = np.sum(target * ln_t_sub_h, axis=1, keepdims=True) / batch

        return kl_div.copy()
        # TODO END

    def backward(self, input, target):
		# TODO START

        input_softmax = softmax(input)
        ret = input_softmax - target
        return ret.copy()
		# TODO END


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START

        batch = input.shape[0]

        input_softmax = softmax(input)

        # target = (batch, 10 = [...t_k])
        # c-e = sum(t_k * log(h_k)) / batch_size
        # (batch, 1)
        cross_entropy = -np.sum(target * np.log(input_softmax), axis=1, keepdims=True) / batch

        return cross_entropy.copy()
        # TODO END

    def backward(self, input, target):
        # TODO START

        input_softmax = softmax(input)
        ret = input_softmax - target
        return ret.copy()
        # TODO END


class HingeLoss(object):
    def __init__(self, name, margin=5):
        self.name = name
        self.margin = margin

    def forward(self, input, target):
        # TODO START 

        batch = input.shape[0]

        # x_t^n
        respective_x_tn = np.sum(input * target, axis=1, keepdims=True)

        # mask if t_n == 1
        mask = np.where(target == 0, 1, 0)

        # hinge = max(0, x_k - x_t^n + delta)
        hinge = np.maximum(0, input - respective_x_tn + self.margin) * mask

        ret = np.sum(hinge, axis=1, keepdims=True) / batch

        return ret.copy()
        # TODO END

    def backward(self, input, target):
        # TODO START

        # x_t^n
        respective_x_tn = np.sum(input * target, axis=1, keepdims=True)

        # whether grad exist
        distance_exist = np.where(input - respective_x_tn + self.margin > 0, 1, 0)

        hinge_grad = distance_exist * (1 - target)

        # for correct labels, subtract, in encouragement of higher score
        # suggestions from ChatGPT-4o

        # prompt: <original code>,
        # for current back propagation code, do you have some suggestions about the right class?

        target_grad = -np.sum(hinge_grad, axis=1, keepdims=True) * target

        ret = hinge_grad + target_grad

        return ret.copy()
        # TODO END


# Bonus
class FocalLoss(object):
    def __init__(self, name, alpha=None, gamma=2.0):
        self.name = name
        if alpha is None:
            self.alpha = [0.1 for _ in range(10)]
        self.gamma = gamma

    def forward(self, input, target):
        # TODO START

        # the same as cross_entropy from observation
        batch = input.shape[0]

        input_softmax = softmax(input)

        cross_entropy = target * np.log(input_softmax)

        # get classification focal correction
        np_alpha = np.array(self.alpha)

        suc_k = np_alpha * target + (1 - np_alpha) * (1 - target)

        focal = cross_entropy * ((1 - input_softmax) ** self.gamma) * suc_k

        ret = -np.sum(focal, axis=1, keepdims=True) / batch

        return ret.copy()
        # TODO END

    def backward(self, input, target):
        # TODO START

        np_alpha = np.array(self.alpha)

        input_softmax = softmax(input)

        suc_k = np_alpha * target + (1 - np_alpha) * (1 - target)
        d_gamma = self.gamma * (1 - input_softmax) ** (self.gamma - 1) * target * np.log(input_softmax)
        d_log = -(1 - input_softmax) ** self.gamma * target * (1 / input_softmax)
        d_out = suc_k * (d_gamma + d_log)

        # d_in = softmax_jacobi shape = (batch, 10, 10)
        batch, channel = input.shape[0], input.shape[1]

        # get jacobi matrix
        softmax_jacobi = np.zeros((batch, channel, channel))
        for i in range(batch):
            p = input_softmax[i].reshape(-1, 1)  # (channel, 1)
            diag = np.diagflat(p)  # (channel, channel)
            softmax_jacobi[i] = diag - (p @ p.T)

        # d_out * d_in, d[i] = jacobi[i] @ d_out[i]
        # (channel, channel) @ (channel, 1) in each batch
        ret = np.matmul(softmax_jacobi, d_out[..., np.newaxis]).squeeze(-1)
        return ret.copy()
        # TODO END