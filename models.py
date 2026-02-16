import numpy as np
from activations import derivatives, ReLU, glorot_uniform, zeros


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
        self.isflat = not activation
        self.n = n
        self.a = None
        if activation:
            self.w = None
            self.b = None
            self.z = np.array([0] * n, dtype=float)
            self.f = activation
            self.f_w = initializer_w
            self.f_b = initializer_b
            self.df = derivatives[activation]
    
    def apply(self, a):
        if self.isflat:
            self.a = a.flatten()
            self.n = a.size
            return self.a
        raise EnvironmentError("Dense layer cannot function in place of a flatten layer.\n Don't specify an activation or a number for the neurons to create a [FlattenLayer]")

class Kernel:
    def __init__(self, size=1, stride=1, padding=0, activation=ReLU, initializer_w=glorot_uniform, initializer_b=zeros):
        self.size = size
        self.stride = stride
        self.padding = padding
        self.w = initializer_w(size, size)
        self.a = None
        self.z = None
        self.b = None
        self.f_b = initializer_b
        self.f = activation
        self.df = derivatives.get(activation)

    def apply(self, a):
        # The length of feature_map:
        # (data.shape[0] - self.size - 2 * self.padding) // self.stride + 1
        size_x, size_y = len(a[0]), len(a)

        # add a padding box - up and down, left and right
        if self.padding:
            a = [[0] * size_x] * self.padding + a + [[0] * size_x] * self.padding
            for i in range(size_y): a[i] = [0] * self.padding + a[i] + [0] * self.padding 

        # transform the data to a numpy array
        a = np.array(a, dtype=float)

        feature_map = []
        # do convolution between the array of data and the kernels' weights
        for i in range(0, size_y - self.size + 1, self.stride):
            for j in range(0, size_x - self.size + 1, self.stride):
                z_dash = a[i : i + self.size, j : j + self.size] * self.w
                feature_map.append(np.sum(z_dash))

        feature_map = np.array(feature_map, dtype=float)
        self.b = self.f_b(*feature_map.shape)
        
        try:
            self.z = feature_map + self.b # add the biases
            self.a = self.f(self.z) # apply non-linearity
            return self.a
        except:
            raise ModuleNotFoundError("There is no non-linearity function given.")

class Channels():
    def __init__(self, n=1, size=1, stride=1, padding=0, activation=ReLU, initializer_w=glorot_uniform, initializer_b=zeros):
        self.args = [
        Kernel(size, stride, padding, lambda x:x, initializer_w, zeros) for _ in range(n)
        ]
        self.a = None
        self.b = None
        self.f_b = initializer_b
        self.f = activation
        self.df = derivatives.get(activation)
    
    @property
    def w(self):
        return np.array([c.w for c in self.args], dtype=float)
    
    @property
    def z(self):
        return np.array([c.z for c in self.args], dtype=float)

    def apply(self, data_channels):
        feature_maps = [kernel.apply(channel) for kernel, channel in zip(self.args, data_channels)]
        
        self.b = self.f_b(*feature_maps[0].shape)
        z = self.b + feature_maps[0]

        for map in feature_maps[1:]: z += map
        
        self.a = self.f(z)
        return self.a

class Branches():
    def __init__(self, *kernels):
        self.args = kernels
    
    @property
    def w(self):
        return np.array([c.w for c in self.args], dtype=float)
    
    @property
    def z(self):
        return np.array([c.z for c in self.args], dtype=float)
    
    def apply(self, data):
        return [kernel.apply(data) for kernel in self.args]

class Pooling():
    def __init__(self, size=1, function="max"):
        self.size = size
        self.f = function
        self.a = None
    
    def apply(self, feature_map):
        size_x, size_y = len(feature_map[0]), len(feature_map)

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
                    z_dash = pooled_map[i : i + self.size, j : j + self.size]
                    
                    if self.f == "max": pooled_map.append(np.max(z_dash))
                    elif self.f == "min": pooled_map.append(np.min(z_dash))
                    elif self.f == "average" or self.f == "avg": pooled_map.append(np.average(z_dash))    
                    else: pooled_map.append(self.f(z_dash))
                    
        self.a = np.array(pooled_map, dtype=float)
        
        return self.a


# Networks
class NeuralNetwork:
    def __init__(self, possible_outcomes=None, *layers, cost=None, dcost=None, optimizer=None):
        self.layers_lengths = [layer.n for layer in layers]
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
            n_in = self.layers_lengths[i]
            n_out = self.layers_lengths[i + 1]
            
            
            l = self.params[i + 1]

            l.w = l.f_w(n_in, n_out)
            l.b = l.f_b(n_out)

    def feedforward(self, a):
        self.params[0].a = a
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
        # Yup. This is so scary (so crazy)
        # nor my (or your) computer will hold on
        
        # So, Newton formula of high schools is safer

        l = self.params[i + 1]
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
        l = self.params[i + 1]

        self.feedforward(x)
        L1 = self.loss(self.output_vector())

        l.b[j] += δ
            
        self.feedforward(x)
        L2 = self.loss(self.output_vector())

        l.b[j] -= δ

        db = L2 - L1 / δ
        return db

    def loss_gradient(self, i, dz):
        l = self.params[i]
        # prev_a = l.a
        dw = np.outer(dz, l.a)
        db = dz
        if i > 0:
            da = self.params[i + 1].w.T @ dz
            dz = da * l.df(l.z)
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
            dw, db, dz = self.loss_gradient(i, dz)
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
                raise ZeroDivisionError(f"batch_size shouldn't be zero.")

            loss_traj.append(loss_percentage)
            accuarcy.append(acc_percentage)

            print(f"accuracy-{i + 1}:", f"{round(acc_percentage, 2)}%")
        return loss_traj, accuarcy
        
    def set_optimizer(self, optimizer=None):
        self.optim = optimizer

    def output_vector(self):
        return self.params[-1].a
    
    def output(self):
        last_layer = self.params[-1].a
        if self.layers_lengths[-1] > 1:
            for node, out in zip(last_layer, self.possible_outcomes):
                if node >= max(last_layer):
                    return out
        else:
            return self.possible_outcomes[1] if self.params[-1].a[0] > 0.5 else self.possible_outcomes[0]

class ConvolutionalNeuralNetwork(NeuralNetwork):
    def __init__(self, possible_outcomes=None, *sequence, cost=None, dcost=None, optimizer=None):
        self.params = [s for s in sequence if (isinstance(s, (Layer, Kernel, Channels, Branches)))]
        
        # take out flatten layers that comes from [Layer] class.
        for param in self.params:
            if isinstance(param, Layer):
                if param.isflat: self.params.remove(param)

        self.length = len(self.params)
        self.possible_outcomes = possible_outcomes
        self.optim = optimizer
        self.c, self.dc = cost, dcost

        # check for automated configuration
        self.sequence = sequence
        for i, s in enumerate(sequence, 1):
            if isinstance(s, (Layer, Flatten)) and isinstance(sequence[i - 1], Kernel):
                self.sequence.insert(i + 1, Flatten())
            elif isinstance(s, Kernel) and isinstance(sequence[i - 1], Layer):
                self.sequence.insert(i + 1, Reshape())

        # inform the optimizer of the CNN's structure by
        # passing the number of layers to the optimizer
        if optimizer != None:
            optimizer.N = self.length - 1

    def feedforward(self, a):
        prev_shape = a.shape       
        s = self.sequence[0]

        if isinstance(s, (Flatten, Layer, Kernel, Channels, Branches, Pooling)): a = s.apply(a)
        else: raise EnvironmentError("Your sturcture should start with Kernel, Channels, Branches, Pooling or Layer.")
        
        for i, s in enumerate(self.sequence[1:]):
            if isinstance(s, (Flatten, Kernel, Channels, Branches, Pooling)) or (isinstance(s, Layer) and s.isflat):
                a = s.apply(a)
                prev_shape = a.shape

            elif isinstance(s, Reshape):
                if s.shape == -1:
                    x, y = prev_shape 
                    ratio = a.size // x * y + 1
                    s.shape = prev_shape = (x * ratio, y * ratio)
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
        l = self.params[i]
        prev_a = l.a
        dw = np.outer(dz, prev_a)
        db = dz
        
        if i > 0:
            if isinstance(l, Kernel):
                da = l.w.T @ dz
            elif isinstance(l, (Channels, Branches)):
                da = np.array([k.w.T @ dz for k in l.args], dtype=float)
            else:
                da = self.params[i + 1].w.T @ dz
            dz = da * l.df(l.z)
        
        return dw, db, dz

    def output_vector(self):
        return self.sequence[-1].a
    
    def output(self):
        last_element = self.sequence[-1].a
        if len(self.params[-1].a) > 1:
            for node, out in zip(last_element, self.possible_outcomes):
                if node >= max(last_element):
                    return out
        else:
            return self.possible_outcomes[1] if self.sequence[-1].a[0] > 0.5 else self.possible_outcomes[0]

# train the hyperparameters 
# 1. kernel's weight, biases
# 2. stride (typically fixed)
