from data_utils import load_data
from NN_model import NNModel


X_train, Y_train, X_test, Y_test = load_data("none")

my_model = NNModel(X_test, Y_test, num_passes=20000, reg_lambda=0.01, epsilon=0.0001)
my_model.build_model(X_train, Y_train, 32)

exit(0)
