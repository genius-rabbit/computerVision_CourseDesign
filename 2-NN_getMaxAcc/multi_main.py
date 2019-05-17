from data_utils import load_data
from NN_Multi_layer_model import NNModel
import matplotlib.pyplot as plt


X_train, Y_train, X_test, Y_test = load_data("none")

my_model = NNModel(X_test, Y_test, num_passes=3000, reg_lambda=0.0, epsilon=0.00001)
my_model.build_model()
losses, acces = my_model.train_model(X_train, Y_train)

plt.errorbar(range(len(losses)), losses)
plt.title('NN classifier')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

plt.errorbar(range(len(acces)), acces)
plt.title('NN classifier')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()

exit()

