import matplotlib.pyplot as plt
import numpy as np
from data_utils import load_data

# import sklearn
# import sklearn.datasets
# import sklearn.linear_model
# import matplotlib
# from pprint import pprint

# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates the contour plot below.
def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=yy, cmap=plt.cm.Spectral)


# Helper function to evaluate the total loss on the dataset
def calculate_loss(model, x, y):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # 正向传播，计算预测值
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # 计算损失
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # 在损失上加上正则项（可选）
    data_loss += reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1. / num_examples * data_loss


# Helper function to predict an output (0 or 1)
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2

    exp_score = np.exp(z2)
    probs = exp_score / np.sum(exp_score, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(x_train, y_train, nn_input_dim, nn_hdim, print_loss=False):
    nn_output_dim = y_train.shape[0]
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros([1, nn_hdim])
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros([1, nn_output_dim])

    my_model = {}

    for i in range(epochs):
        # forward
        print(b1.shape, x.dot(W1).shape)
        z1 = x_train.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        delta3 = probs
        delta3[range(num_examples), y_train] -= 1
        dw2 = a1.T.dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)

        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dw1 = np.dot(x_train.T, delta2)
        db1 = np.sum(delta2, axis=0)

        dw2 += reg_lambda * dw2
        dw1 += reg_lambda * dw1

        W1 += -epsilon * dw1
        b1 += -epsilon * db1
        W2 += -epsilon * dw2
        b2 += -epsilon * db2

        my_model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        if print_loss and i % 1000 == 0:
            print("Loss after iteration %i : %f" % (i, calculate_loss(model, x, y)))

    return my_model


# load data
X_train, Y_train, X_test, Y_test = load_data()

# Build a model with a 3-dimensional hidden layer

num_examples, input_dim = X_train.shape
epsilon = 0.001
reg_lambda = 0.00
epochs = 1

model = build_model(X_train, Y_train, nn_input_dim=input_dim, nn_hdim=100, print_loss=True)

n_correct = 0
n_test = X_test.shape[0]
for n in range(n_test):
    x = X_test[n, :]
    yp = predict(model, x)
    if yp == Y_test[n]:
        n_correct += 1.0

print('Accuracy %f = %d / %d' % (n_correct / n_test, int(n_correct), n_test))

exit(0)
