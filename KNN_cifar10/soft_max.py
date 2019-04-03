import numpy as np
import random


class SoftMax(object):
    def __init__(self, batch_size, learn_rate, epoch):
        self.weights = None
        self.epoch = epoch
        self.batchSize = batch_size
        self.learnRate = learn_rate

    def predict_acc(self, x_test, y_test):
        score = np.dot(x_test, self.weights)
        y_pred = np.zeros(score.shape[0])
        for index in range(score.shape[0]):
            y_pred[index] = np.argmax(score[index])

        acc = np.sum(y_pred == y_test) / float(y_test.shape[0])
        return acc

    '''
    if i==label
        dW = xj*pi - xj = xj(pi - 1)
    else
        dW = xj*pi
    '''
    def compute_loss(self, x, y):
        batch_size = x.shape[0]
        score = np.dot(x, self.weights)
        score -= score.max()

        softmax = np.exp(score)
        softmax /= softmax.sum(axis=1).reshape([-1, 1])
        loss = - np.log(softmax[np.arange(len(softmax)), y]).sum()
        loss /= batch_size

        softmax[np.arange(len(softmax)), y] -= 1
        dW = x.T.dot(softmax)

        return loss, dW

    def evaluate_gradient(self, data_x, data_y):
        losses = []
        randomIndices = random.sample(range(data_x.shape[0]), data_x.shape[0])
        data_x = data_x[randomIndices]
        data_y = data_y[randomIndices]
        for index in range(0, data_x.shape[0], self.batchSize):
            x_mini_batch = data_x[index: index + self.batchSize]
            y_mini_batch = data_y[index: index + self.batchSize]

            loss, dW = self.compute_loss(x_mini_batch, y_mini_batch)
            self.weights -= self.learnRate * dW
            losses.append(loss)

        return np.sum(losses) / len(losses)

    def train(self, x_train, y_train, x_test, y_test):
        n_features = x_train.shape[1]
        n_classes = y_train.max() + 1
        self.weights = np.random.randn(n_features, n_classes)
        train_losses = []
        test_losses = []
        test_acces = []
        for index in range(self.epoch):
            train_loss = self.evaluate_gradient(x_train, y_train)

            test_loss, _ = self.compute_loss(x_test, y_test)
            test_acc = self.predict_acc(x_test, y_test)

            test_acces.append(test_acc)
            test_losses.append(test_loss)
            train_losses.append(train_loss)
            print('epoch %d  train_loss: %.6f test_loss: %.6f acc: %.6f' % (index, train_loss, test_loss, test_acc))

        return test_acces, test_losses, train_losses
