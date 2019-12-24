import numpy as np
from nn import NN
import matplotlib.pyplot as plt

n_epochs = 50
seed = 0

nn = NN(hidden_dims=(512, 120, 120, 120, 120, 120, 120), 
    lr=0.003, batch_size=100, seed=seed,
    activation='relu')
train_logs = nn.train_loop(n_epochs)

plt.plot(np.arange(n_epochs), train_logs['train_accuracy'], label='train')
plt.plot(np.arange(n_epochs), train_logs['validation_accuracy'], label='validation')

plt.xlabel("epochs")
plt.ylabel("accuracy")

plt.legend(loc="upper left")
plt.title("model accuracy")
plt.savefig("output_acc.png")

plt.close()

plt.plot(np.arange(n_epochs), train_logs['train_loss'], label='train')
plt.plot(np.arange(n_epochs), train_logs['validation_loss'], label='validation')

plt.xlabel("epochs")
plt.ylabel("loss")

plt.legend(loc="upper left")
plt.title("model loss")
plt.savefig("output_loss.png")

plt.close()