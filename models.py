import numpy as np
from activations import derivatives, ReLU, glorot_uniform, zeros, zeros_map

class Reshape():
    def __init__(self, shape=-1):
        self.shape = shape
    
    def apply(self, a):
        self.a = a.reshape(self.shape)
        return self.a

class Flatten():
    def __init__(self):
        self.n = 0
        self.a = None
    
    def apply(self, a):
        self.a = a.flatten()
        self.n = a.size
        return self.a
    
class Layer():
    def __init__(self, n=0, activation=None, initializer_w=glorot_uniform, initializer_b=zeros):
        self.n = n
        self.a = None
        if activation:
            self.w = None
            self.b = None
            self.z = np.array([0] * n, dtype=float)
            self.f = activation
            self.df = derivatives[activation]
            self.f_w = initializer_w
            self.f_b = initializer_b
    
class ConvLayer:
    def __init__(self, branches=1, depth=1, size=1, stride=1, padding=0, activation=ReLU, initializer_w=glorot_uniform, initializer_b=zeros_map):
        self.branches = branches
        self.depth = depth
        self.size = size
        self.stride = stride
        self.padding = padding
        self.kernels = [[initializer_w(size, size) for _ in range(depth)] for _ in range(branches)]
        self.f_b = initializer_b
        self.b = None
        self.z = []

    def apply(self, data_channels):
        x = (data_channels[0].shape[0] - self.size - 2 * self.padding) // self.stride + 1
        y = (data_channels[0].shape[1] - self.size - 2 * self.padding) // self.stride + 1
        
        self.b = [self.f_b(x, y) for _ in range(self.branches)]
        
        for k, b in zip(self.kernels, self.b):
            z = b
            for w, a in zip(k, data_channels):
                size_x, size_y = a.shape

                # add a padding box - up and down, left and right
                if self.padding:
                    a = np.concatenate(([[0] * size_x] * self.padding, a))
                    a = np.concatenate((a, [[0] * size_x] * self.padding))
                    
                    size_y += self.padding * 2
                    
                    a = np.concatenate(([[0] * self.padding] * size_y, a), axis=1)
                    a = np.concatenate((a, [[0] * self.padding] * size_y), axis=1) 

                # transform the data to a numpy array
                a = np.array(a, dtype=float)

                feature_map = []
                # do convolution between the array of data and the kernels' weights
                for i in range(0, size_y - self.size + 1, self.stride):
                    feature_map.append([])
                    for j in range(0, size_x - self.size + 1, self.stride):
                        z_dash = a[i : i + self.size, j : j + self.size] * w
                        feature_map[-1].append(np.sum(z_dash))

                feature_map = np.array(feature_map, dtype=float)
                
                z += feature_map
            
            self.z.append(z)
        try:
            self.a = self.f(self.z) # apply non-linearity
        except:
            raise ModuleNotFoundError("There is no non-linearity function given.")

class Pooling():
    def __init__(self, size=1, function="max"):
        self.size = size
        self.f = function
        self.a = None
    
    def apply(self, feature_map):
        size_x, size_y = feature_map.shape

        # transform the data to a numpy array
        feature_map = np.array(feature_map, dtype=float)

        pooled_map = []

        # do pooling for the feature map
        if self.f in ["globalmax", "global_max", "gmax"]:
            pooled_map = np.max(feature_map)
        elif self.f in ["globalmin", "global_min", "gmin"]:
            pooled_map = np.min(feature_map)
        elif self.f in ["globalavg", "global_avg", "gavg"]:
            pooled_map = np.average(feature_map)
        else:
            for i in range(0, size_y - self.size + 1):
                for j in range(0, size_x - self.size + 1):
                    z_dash = feature_map[i : i + self.size, j : j + self.size]
                    
                    if self.f == "max": pooled_map.append(np.max(z_dash))
                    elif self.f == "min": pooled_map.append(np.min(z_dash))
                    elif self.f == "average" or self.f == "avg": pooled_map.append(np.average(z_dash))    
                    else: pooled_map.append(self.f(z_dash))
                    
        self.a = np.array(pooled_map, dtype=float)
        
        return self.a


# Networks
class NeuralNetwork:
    def __init__(self, *layers, possible_outcomes=None, cost=None, dcost=None, optimizer=None):
        self.length = len(layers)
        self.params = layers
        self.possible_outcomes = possible_outcomes
        self.optim = optimizer
        self.c, self.dc = cost, dcost

        # inform the optimizer of the NN's structure by
        # passing the number of layers to the optimizer
        if optimizer != None:
            optimizer.N = self.length - 1

        # initialize weights and baises
        for i in range(self.length - 1):
            n_in = self.params[i].n
            n_out = self.params[i + 1].n
            
            
            l = self.params[i + 1]

            l.w = l.f_w(n_in, n_out)
            l.b = l.f_b(n_out)

    def feedforward(self, a):
        self.params[0].a = a.flatten()
        for l in self.params[1:]:
            z = l.w @ a + l.b
            a = l.f(z)

            l.z = z
            l.a = a
        return a
    
    def loss(self, y):
        return self.c(self.output_vector(), y)

    def approx_gradient_w(self, i, j, k, x, δ=0.001):
        # calculating the gradient with respect to a particular weight would be like:      
        # L(w0, ..., wj, ..., wk, ..., y) = C((f(f(f(...)*wj + ... + b)*wk + ... + b) + (...) - y)
        # Yup. This is so scary (and so crazy)
        # nor my (or your) computer will hold on
        
        # So, Newton formula from high school is safer
        if i <=0:        
            raise IndexError(f"There is no learnable parameter at ({i}) index.")
        
        l = self.params[i]
        
        self.feedforward(x)
        L1 = self.loss(self.output_vector())

        l.w[j][k] += δ
        
        self.feedforward(x)
        L2 = self.loss(self.output_vector())

        l.w[j][k] -= δ

        dw = L2 - L1 / δ

        return dw
    
    def approx_gradient_b(self, i, j, x, δ=0.001):
        # calculate the gradient with respect to a bias
        
        if i <= 0:
            raise IndexError(f"There is no learnable parameter at ({i}) index.")

        l = self.params[i]

        self.feedforward(x)
        L1 = self.loss(self.output_vector())

        l.b[j] += δ
            
        self.feedforward(x)
        L2 = self.loss(self.output_vector())

        l.b[j] -= δ

        db = L2 - L1 / δ
        return db

    def loss_gradient(self, i, dz):
        prev_l = self.params[i - 1]
        # prev_a = l.a
        dw = np.outer(dz, prev_l.a)
        db = dz
        if i > 1:
            da = self.params[i].w.T @ dz
            dz = da * prev_l.df(prev_l.z)
        return dw, db, dz

    def backprop(self, y):
        # This takes derivitative of the cost of a particular datapoint with respect to 
        # one of neurons in the last layer. 
        
        dW, dB = [None] * (self.length - 1), [None] * (self.length - 1)
        
        # initialize dz from the last layer
        if self.dc == None:
            try:
                dz = derivatives[self.c](self.params[-1], y)
            except:
                raise ValueError(f"No differential expression is assigned to ({self.c.__name__})")
        else:
            dz = self.dc(self.params[-1], y) * self.df(self.params[-1].z)

        for i in reversed(range(self.length - 1)):
            dw, db, dz = self.loss_gradient(i + 1, dz)
            if self.optim != None:
                dw, db = self.optim.func(dw, db, i)

            dW[i] = dw
            dB[i] = db
            
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
                    l = self.params[i + 1]
                    
                    if self.optim == None:
                        # normal SGD
                        l.w -= gW[i] * lr
                        l.b -= gB[i] * lr
                    else:
                        l.w -= gW[i]
                        l.b -= gB[i]

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
                raise ZeroDivisionError(f"(batch_size) should not be zero.")

            loss_traj.append(loss_percentage)
            accuarcy.append(acc_percentage)

            print(f"accuracy-{i + 1}:", f"{round(acc_percentage, 2)}%")
        return loss_traj, accuarcy
        
    def set_optimizer(self, optimizer=None):
        self.optim = optimizer

    def output_vector(self):
        return self.params[-1].a
    
    def output(self):
        last = self.params[-1].a
        if self.params[-1].n > 1:
            for node, out in zip(last, self.possible_outcomes):
                if node >= max(last):
                    return out
        else:
            return self.possible_outcomes[1] if self.params[-1].a[0] > 0.5 else self.possible_outcomes[0]

class ConvolutionalNeuralNetwork(NeuralNetwork):
    def __init__(self, *sequence, input_dim=2, possible_outcomes=None, cost=None, dcost=None, optimizer=None):
        self.input_dim = input_dim
        self.possible_outcomes = possible_outcomes
        self.params = [s for s in sequence if (isinstance(s, (Layer, ConvLayer)))]
        self.length = len(self.params)
        self.optim = optimizer
        self.c, self.dc = cost, dcost

        # check for automated configuration
        self.sequence = sequence
        for i, s in enumerate(sequence, 1):
            if isinstance(s, (Layer, Flatten)) and isinstance(sequence[i - 1], (ConvLayer, Pooling)):
                self.params.insert(i + 1, Flatten())
            elif isinstance(s, (ConvLayer, Pooling)) and isinstance(sequence[i - 1], (Layer, Flatten)):
                self.params.insert(i + 1, Reshape())

        # inform the optimizer of the CNN's structure by
        # passing the number of layers to the optimizer
        if optimizer != None:
            optimizer.N = self.length - 1

    def feedforward(self, a):
        prev_shape = a.shape       
        s = self.sequence[0]

        if isinstance(s, (Flatten, Pooling)):
            a = s.apply(a)
        
        elif isinstance(s, Layer):
            s.a = a.flatten()
        
        elif isinstance(s, ConvLayer):
            if a.ndim != self.input_dim + 1:
                a = s.apply([a])
        
        else: raise EnvironmentError("Your sturcture should start with ConvLayer, Layer, or Pooling.")
        
        for i, s in enumerate(self.sequence[1:]):
            if isinstance(s, (Flatten, Pooling)):
                a = s.apply(a)
                prev_shape = a.shape

            elif isinstance(s, ConvLayer):
                if a.ndim != self.input_dim + 1:
                    a = s.apply([a])

            elif isinstance(s, Reshape):
                if s.shape == -1:
                    size = 1
                    for x in prev_shape: size *= x
                    
                    ratio = a.size // size + 1
                    s.shape = prev_shape = (x * ratio for x in prev_shape)
                a = s.apply(a)
            
            elif isinstance(s, Layer):
                # initialize weights and baises
                if s.w is None:
                    n_in = self.sequence[i].n
                    n_out = s.n

                    # register these new pramaters
                    s.w, s.b = s.f_w(n_in, n_out), s.f_b(n_out)
                
                # calculate activations
                z = s.w @ a + s.b

                a = s.f(z)
                s.z = z
                s.a = a
        return a
    
    def loss_gradient(self, i, dz):
        p = self.params[i]
        prev_p = self.params[i - 1]
        dw = np.outer(dz, prev_p.a)
        db = dz
        
        if i > 0:
            if isinstance(p, ConvLayer):
                da = [[w.T @ dz for w in k] for k in p.kernels]
            else:
                da = p.w.T @ dz
            dz = da * prev_p.df(prev_p.z)
        return dw, db, dz

    def output_vector(self):
        return self.params[-1].a
    
    def output(self):
        last = self.params[-1].a
        if len(last) > 1:
            for node, out in zip(last, self.possible_outcomes):
                if node >= max(last):
                    return out
        else:
            return self.possible_outcomes[1] if self.params[-1].a[0] > 0.5 else self.possible_outcomes[0]

# 1. do fast convolution
# 2. train kernel's weight, biases