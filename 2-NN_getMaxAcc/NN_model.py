import numpy as np
import random


class NNModel(object):
    def __init__(self, x_test, y_test, num_passes, reg_lambda, epsilon):
        self.model = {}
        self.num_passes = num_passes
        self.reg_lambda = reg_lambda
        self.epsilon = epsilon
        self.num_examples = 0
        self.x_test = x_test
        self.y_test = y_test

    def predict(self, x, y):
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']

        z1 = x.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2

        exp_score = np.exp(z2)
        probs = exp_score / np.sum(exp_score, axis=1, keepdims=True)
        pred = np.argmax(probs, axis=1)
        acc = np.sum(pred == y) / float(y.shape[0])
        return acc

    def calculate_loss(self, x, y):
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        # 正向传播，计算预测值
        z1 = x.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # 计算损失
        correct_logprobs = -np.log(probs[range(self.num_examples), y])
        data_loss = np.sum(correct_logprobs)
        # 在损失上加上正则项（可选）
        data_loss += self.reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        return 1. / self.num_examples * data_loss

    def build_model(self, x_train, y_train, nn_hdim):
        num_examples, nn_input_dim = x_train.shape
        self.num_examples = num_examples
        nn_output_dim = max(y_train) + 1

        W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
        b1 = np.zeros((1, nn_hdim))
        W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
        b2 = np.zeros((1, nn_output_dim))

        losses = np.zeros(int(self.num_passes / 100))
        for i in range(self.num_passes):
            randomIndices = random.sample(range(x_train.shape[0]), x_train.shape[0])
            x_train = x_train[randomIndices]
            y_train = y_train[randomIndices]

            # forward
            z1 = x_train.dot(W1) + b1

            # Leaky ReLU
            # a1 = np.maximum(0.001 * z1, z1)
            # tanh
            a1 = np.tanh(z1)

            z2 = a1.dot(W2) + b2
            exp_scores = np.exp(z2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # back up
            delta3 = probs

            delta3[range(self.num_examples), y_train] -= 1
            dw2 = a1.T.dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            print(a1.shape)
            delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
            dw1 = x_train.T.dot(delta2)
            db1 = np.sum(delta2, axis=0)

            dw2 += self.reg_lambda * dw2
            dw1 += self.reg_lambda * dw1

            W1 += -self.epsilon * dw1
            b1 += -self.epsilon * db1
            W2 += -self.epsilon * dw2
            b2 += -self.epsilon * db2

            self.model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

            if i % 100 == 0:
                index = int(i/100)
                train_loss = self.calculate_loss(x_train, y_train)
                losses[index] = train_loss
                if index >= 3 and losses[index - 3] < losses[index]:
                    break
                print("iteration %i Loss = %f Train Acc = %f  Test Acc = %f"
                      % (i, train_loss,
                         self.predict(x_train, y_train),
                         self.predict(self.x_test, self.y_test)))

        return self.model
