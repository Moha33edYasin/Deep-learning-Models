import numpy as np
from .activitions import derivatives, glorot_uniform, zeros

class Layer:
    def __init__(self, n, activition=None, initializer_w=glorot_uniform, initializer_b=zeros):
        self.n = n
        self.a = np.array([0] * n, dtype=float)
        if activition:
            self.z = np.array([0] * n, dtype=float)
            self.f = activition
            self.f_w = initializer_w
            self.f_b = initializer_b
            self.df = derivatives[activition]

class NeuralNetwork:
    def __init__(self, possible_outcomes=None, *layers, cost=None, dcost=None, optimizer=None):
        self.weights, self.biases = [], []
        self.layers_lengths = [layer.n for layer in layers]
        self.possible_outcomes = possible_outcomes
        self.length = len(layers)
        self.layers = layers
        self.optim = optimizer
        self.c, self.dc = cost, dcost

        # inform the optimizer of the nn's structure by
        # passing the number of layers to the optimizer
        if optimizer != None:
            optimizer.N = self.length - 1

        # initialize weights and baises
        for i in range(self.length - 1):
            n_in = self.layers_lengths[i]
            n_out = self.layers_lengths[i + 1]
            
            f_w, f_b = layers[i + 1].f_w, layers[i + 1].f_b

            self.weights.append(f_w(n_in, n_out))
            self.biases.append(f_b(n_out))

    def feedforward(self, a):
        self.layers[0].a = a
        for i, (w, b) in enumerate(zip(self.weights, self.biases), 1):
            z = w @ a + b
            a = self.layers[i].f(z)

            self.layers[i].z = z
            self.layers[i].a = a
        return a
    
    def loss(self, y):
        return self.c(self.output_vector(), y)

    def approx_gradient_w(self, l, j, k, x, δ=0.001):
        # calculate the gradient with respect to a weight      
        # L(w0, ..., wj, ..., wk, ..., y) = C((f(f(f(...)*wj + ... + b)*wk + ... + b) + (...) - y)
        # this is so scary too derviate
        # nor the computer will hold on
        
        # So, highschool Newton formula is safer
        self.feedforward(x)
        L1 = self.loss(self.output_vector())

        self.weights[l][j][k] += δ
        
        self.feedforward(x)
        L2 = self.loss(self.output_vector())

        self.weights[l][j][k] -= δ

        dw = L2 - L1 / δ

        return dw
    
    def approx_gradient_b(self, l, j, x, δ=0.001):
        # calculate the gradient with respect to a bias        
        self.feedforward(x)
        L1 = self.loss(self.output_vector())

        self.biases[l][j] += δ
            
        self.feedforward(x)
        L2 = self.loss(self.output_vector())

        self.biases[l][j] -= δ

        db = L2 - L1 / δ
        return db

    def loss_gradient(self, l, dz):
        prev_a = self.layers[l].a
        dw = np.outer(dz, prev_a)
        db = dz
        if l > 0:
            da = self.weights[l].T @ dz
            z = self.layers[l].z
            dz = da * self.layers[l].df(z)
        return dw, db, dz

    def backprop(self, y):
        # This takes derivitative of the cost of a particular datapoint with respect to 
        # one of the last layer neurons. 
        
        dW, dB = [None] * (self.length - 1), [None] * (self.length - 1)
        
        # initialize dz from the last layer
        if self.dc == None:
            try:
                dz = derivatives[self.c](self.layers[-1], y)
            except:
                raise ValueError(f"No derivative is assigned to ({self.c.__name__})")
        else:
            dz = self.dc(self.layers[-1], y) * self.df(self.layers[-1].z)

        for l in reversed(range(self.length - 1)):
            dw, db, dz = self.loss_gradient(l, dz)
            if self.optim != None:
                dw, db = self.optim.func(dw, db, l)

            dW[l] = dw
            dB[l] = db
            
        return dW, dB 

    def learn(self, x_data, y_data, targets, lr, epochs=1, batch_size=1):
        loss_traj, accuracy = [], []

        data = list(zip(x_data, y_data, targets))
        N = len(data)
        
        for epoch in range(epochs):
            # backpropagation
            gW, gB = 0, 0
            n_correct, loss = 0, 0
            
            for i in range(0, N, batch_size):
                batch = data[i : batch_size + i]
                
                # shuffle the batch
                idx = np.random.permutation(len(batch))
                batch = [batch[i] for i in idx]

                for x, y, t in batch:
                    self.feedforward(x)
                    loss += self.loss(y)
                    dW, dB = self.backprop(y)
                    
                    gW = [gw + dw for gw, dw in zip(gW, dW)] if gW else dW.copy()
                    gB = [gb + db for gb, db in zip(gB, dB)] if gB else dB.copy()

                    n_correct += int(self.output() == t)
                
                gW = [gw / batch_size for gw in gW]
                gB = [gb / batch_size for gb in gB]

                # update gradient
                for i in reversed(range(self.length - 1)):
                    if self.optim == None:
                        # normal SGD
                        self.weights[i] -= gW[i] * lr
                        self.biases[i] -= gB[i] * lr
                    else:
                        self.weights[i] -= gW[i]
                        self.biases[i] -= gB[i]


            # calculate loss and accuracy per batch
            try:
                acc_percentage = np.round(n_correct / N * 100, 2)
                loss_percentage = np.round(loss / N * 100, 2)
            except:
                raise ZeroDivisionError(f"batch_size shouldn't be zero.")

            loss_traj.append(loss_percentage)
            accuracy.append(acc_percentage)
            print(f"accuracy-{epoch + 1}:", f"{acc_percentage}%")

        return loss_traj, accuracy

    def test(self, x_data, y_data, targets, batch_size=1):
        loss_traj, accuarcy = [], []
        data = list(zip(x_data, y_data, targets))
        for i in range(0, len(x_data), batch_size):
            loss, n_correct = 0, 0
            for x, y, t in data[i : batch_size + i]:
                self.feedforward(x)
                loss += self.loss(y)
                n_correct += int(self.output() == t)
            
            # calculate loss and accuarcy per batch
            try:
                acc_percentage = np.round(n_correct / batch_size * 100, 2)
                loss_percentage = np.round(loss / batch_size * 100, 2)
            except:
                raise ZeroDivisionError(f"batch_size shouldn't be zero.")

            loss_traj.append(loss_percentage)
            accuarcy.append(acc_percentage)

            print(f"accuracy-{i + 1}:", f"{round(acc_percentage, 2)}%")
        return loss_traj, accuarcy
        
    def set_optimizer(self, optimizer=None):
        self.optim = optimizer

    def output_vector(self):
        return self.layers[-1].a
    
    def output(self):
        last_layer = self.layers[-1].a
        m = max(last_layer)
        if self.layers_lengths[-1] > 1:
            for node, out in zip(last_layer, self.possible_outcomes):
                if node >= m:
                    return out
        else:
            return self.possible_outcomes[1] if self.layers[-1].a[0] > 0.5 else self.possible_outcomes[0]