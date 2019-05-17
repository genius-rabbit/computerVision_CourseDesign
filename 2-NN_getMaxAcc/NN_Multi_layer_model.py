import numpy as np
import random


class NNModel(object):
    def __init__(self, x_test, y_test, num_passes, reg_lambda, epsilon):
        self.num_passes = num_passes
        self.reg_lambda = reg_lambda
        self.epsilon = epsilon
        self.num_examples = 0
        self.x_test = x_test
        self.y_test = y_test
        self.layer = [1024, 256, 4]
        self.W = []
        self.b = []
        self.backup = 'tanh'

    def predict(self, x, y):
        for m in range(self.layer.__len__() - 1):
            if m != 0:
                x = np.tanh(x)
            x = x.dot(self.W[m]) + self.b[m]

        exp_score = np.exp(x)
        probs = exp_score / np.sum(exp_score, axis=1, keepdims=True)
        pred = np.argmax(probs, axis=1)
        acc = np.sum(pred == y) / float(y.shape[0])
        return acc

    def calculate_loss(self, x, y):
        # 正向传播，计算预测值
        w_sum = 0
        for m in range(self.layer.__len__() - 1):
            if m != 0:
                x = np.tanh(x)
            x = x.dot(self.W[m]) + self.b[m]
            w_sum += np.sum(np.square(self.W[m]))

        exp_scores = np.exp(x)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # 计算损失
        correct_logprobs = -np.log(probs[range(self.num_examples), y])
        data_loss = np.sum(correct_logprobs)
        # 在损失上加上正则项（可选）
        data_loss += self.reg_lambda / 2 * w_sum
        return 1. / self.num_examples * data_loss

    def build_model(self):
        for i in range(self.layer.__len__() - 1):
            self.W.append(np.random.randn(self.layer[i], self.layer[i+1]) / np.sqrt(self.layer[i]))
            self.b.append(np.zeros((1, self.layer[i+1])))
            print("layer: %d " % i, self.W[i].shape)

    def train_model(self, x_train, y_train):

        # losses = np.zeros(int(self.num_passes / 100))
        # acces = np.zeros(int(self.num_passes / 100))
        losses = []
        acces = []
        for i in range(self.num_passes):
            num_examples, nn_input_dim = x_train.shape
            self.num_examples = num_examples

            # random data set
            randomIndices = random.sample(range(x_train.shape[0]), x_train.shape[0])
            x_train = x_train[randomIndices]
            y_train = y_train[randomIndices]

            # forward
            x = x_train
            a = []
            for m in range(self.layer.__len__() - 1):
                if m == 0:
                    a.append(x)
                else:
                    if self.backup == 'relu':
                        x = np.maximum(0, x)
                    else:
                        x = np.tanh(x)
                    a.append(x)
                x = x.dot(self.W[m]) + self.b[m]
            # print("x shape", x.shape)
            exp_scores = np.exp(x)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # back up
            my_delta = probs
            # print("delta shape:", my_delta.shape)
            delta = []
            for m in range(self.layer.__len__() - 1):
                right = self.layer.__len__() - 1 - m
                if m == 0:
                    my_delta[range(self.num_examples), y_train] -= 1
                else:
                    # print(m, right, my_delta.shape, self.W[right].shape, a[m].shape)
                    if self.backup == 'relu':
                        my_delta = my_delta.dot(self.W[right].T) * (a[right] != 0)
                    else:
                        my_delta = my_delta.dot(self.W[right].T) * (1 - np.power(a[right], 2))

                delta.append(my_delta)

            for m in range(a.__len__()):
                right = a.__len__() - 1 - m
                dW = a[m].T.dot(delta[right])
                db = np.sum(delta[right], axis=0)
                dW += self.reg_lambda * dW

                self.W[m] += -self.epsilon * dW
                self.b[m] += -self.epsilon * db

            nums_epoch = 50
            if i % nums_epoch == 0:
                index = int(i/nums_epoch)
                train_loss = self.calculate_loss(x_train, y_train)
                losses.append(train_loss)

                if index >= 3 and losses[index - 3] < losses[index]:
                    print("loss is up up !!!")
                    # break

                acc = self.predict(self.x_test, self.y_test)
                acces.append(acc)
                print("iteration %i Loss = %f Train Acc = %f  Test Acc = %f"
                      % (i, train_loss,
                         self.predict(x_train, y_train),
                         acc))
        return losses, acces
