import pickle
import numpy as np

class NN(object):
    def __init__(self,
                 hidden_dims=(512, 256),
                 datapath='cifar10.pkl',
                 n_classes=10,
                 epsilon=1e-6,
                 lr=7e-4,
                 batch_size=1000,
                 seed=None,
                 activation="relu",
                 init_method="glorot"
                 ):

        self.hidden_dims = hidden_dims
        self.n_hidden = len(hidden_dims)
        self.datapath = datapath
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.init_method = init_method
        self.seed = seed
        self.activation_str = activation
        self.epsilon = epsilon

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}

        if datapath is not None:
            u = pickle._Unpickler(open(datapath, 'rb'))
            u.encoding = 'latin1'
            self.train, self.valid, self.test = u.load()
        else:
            self.train, self.valid, self.test = None, None, None

    def initialize_weights(self, dims):
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = {}
        # self.weights is a dictionary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):
            # Xavier initialization
            sqrt_inv = 1./np.sqrt(all_dims[layer_n-1])
            self.weights[f"W{layer_n}"] = np.random.uniform(-sqrt_inv, sqrt_inv, size=(all_dims[layer_n-1], all_dims[layer_n]))
            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))

    def relu(self, x, grad=False):
        if grad:
            return (x > 0) * 1
        return np.clip(x, 0, None)

    def sigmoid(self, x, grad=False):
        f = 1. / (1 + np.exp(-x))
        if grad:
            return f * (1 - f)
        return f

    def tanh(self, x, grad=False):
        f = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        if grad:
            return 1 - f**2
        return f

    def activation(self, x, grad=False):
        if self.activation_str == "relu":
            return self.relu(x, grad)
        elif self.activation_str == "sigmoid":
            return self.sigmoid(x, grad)
        elif self.activation_str == "tanh":
            return self.tanh(x, grad)
        else:
            raise Exception("invalid")
        return 0

    def softmax(self, x):
        # Remember that softmax(x-C) = softmax(x) when C is a constant.
        if x.ndim == 1:
            shifted_x = x - np.max(x)
            e_x = np.exp(shifted_x)
            return e_x / np.sum(e_x)
        elif x.ndim == 2:
            shifted_x = x - np.max(x, axis=1, keepdims=True)
            e_x = np.exp(shifted_x)
            return e_x / np.sum(e_x, axis=1, keepdims=True)

    def forward(self, x):
        cache = {"Z0": x}
        # cache is a dictionary with keys Z0, A1, ..., Zm, Am where m - 1 is the number of hidden layers
        # Ai corresponds to the preactivation at layer i, Zi corresponds to the activation at layer i
        
        N_layers = self.n_hidden + 1
        for layer_n in range(1, N_layers):
            cache[f"A{layer_n}"] = np.dot(cache[f"Z{layer_n-1}"], self.weights[f"W{layer_n}"]) + self.weights[f"b{layer_n}"]
            cache[f"Z{layer_n}"] = self.activation(cache[f"A{layer_n}"])
        
        cache[f"A{N_layers}"] = np.dot(cache[f"Z{N_layers-1}"], self.weights[f"W{N_layers}"]) + self.weights[f"b{N_layers}"]
        cache[f"Z{N_layers}"] = self.softmax(cache[f"A{N_layers}"])

        return cache
        
    def backward(self, cache, labels):
        N_layers = self.n_hidden + 1
        output = cache[f"Z{N_layers}"]
        grads = {}
        # grads is a dictionary with keys dAm, dWm, dbm, dZ(m-1), dA(m-1), ..., dW1, db1
        
        grad_oa = grads[f"dA{N_layers}"] = output - labels
        grads[f"dW{N_layers}"] = np.dot(cache[f"Z{N_layers - 1}"].T, grad_oa)
        grads[f"db{N_layers}"] = grad_oa

        for layer_n in range(N_layers-1, 0, -1):
            grads[f"dZ{layer_n}"] = np.dot(grads[f"dA{layer_n + 1}"], self.weights[f"W{layer_n + 1}"].T)
            grads[f"dA{layer_n}"] = grads[f"dZ{layer_n}"] * self.activation(cache[f"A{layer_n}"], grad=True)
            grads[f"dW{layer_n}"] = np.dot(cache[f"Z{layer_n - 1}"].T, grads[f"dA{layer_n}"])
            grads[f"db{layer_n}"] = grads[f"dA{layer_n}"]

        batch_size, _ = output.shape

        for layer in range(1, self.n_hidden + 2):
            grads[f"dW{layer}"] /= batch_size
            grads[f"db{layer}"] = np.mean(grads[f"db{layer}"], axis=0, keepdims=True)

        return grads

    def update(self, grads):
        for layer in range(1, self.n_hidden + 2):
            self.weights[f"W{layer}"] -= self.lr * grads[f"dW{layer}"]
            self.weights[f"b{layer}"] -= self.lr * grads[f"db{layer}"]       

    def one_hot(self, y):
        out = np.eye(self.n_classes)[y]
        return out
        
    def loss(self, prediction, labels):
        prediction[np.where(prediction < self.epsilon)] = self.epsilon
        prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon
        ce = -np.sum(labels * np.log(prediction)) / prediction.shape[0]
        return ce

    def compute_loss_and_accuracy(self, X, y):
        one_y = self.one_hot(y)
        cache = self.forward(X)
        predictions = np.argmax(cache[f"Z{self.n_hidden + 1}"], axis=1)
        accuracy = np.mean(y == predictions)
        loss = self.loss(cache[f"Z{self.n_hidden + 1}"], one_y)
        return loss, accuracy, predictions

    def train_loop(self, n_epochs):
        X_train, y_train = self.train
        y_onehot = self.one_hot(y_train)
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]

                cache = self.forward(minibatchX)
                grads = self.backward(cache, minibatchY)
                self.update(grads)

            X_train, y_train = self.train
            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
            X_valid, y_valid = self.valid
            valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)

            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['validation_accuracy'].append(valid_accuracy)
            self.train_logs['train_loss'].append(train_loss)
            self.train_logs['validation_loss'].append(valid_loss)

            print(f'epoch: {epoch}')
            print(f'train_loss: {train_loss}, train_acc: {train_accuracy}')
            print(f'validation_loss: {valid_loss}, validation_acc: {valid_accuracy}\n')

        return self.train_logs

    def evaluate(self):
        X_test, y_test = self.test
        loss, accuracy, _ = self.compute_loss_and_accuracy(X_test, y_test)
        return loss, accuracy
