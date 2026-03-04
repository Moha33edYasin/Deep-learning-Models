import numpy as np
from methods import derivatives, ReLU, glorot_uniform, zeros, zeros_map

class Reshape():
    def __init__(self, shape=-1):
        self.shape = shape
        self.original_shape = None
        self.a = np.array([])
        self.z = np.array([])
    
    def forward_pass(self, a, z=np.empty((1,))):
        self.original_shape = a.shape
        self.a = a.reshape(self.shape)
        self.z = z.reshape(self.shape)
        return self.a

    def backward_pass(self, dz):
        return dz.reshape(self.original_shape)

class Flatten():
    def __init__(self):
        self.original_shape = None
        self.n = 0
        self.a = np.array([])
        self.z = np.array([])
    
    def forward_pass(self, a, z=np.empty((1,))):
        self.original_shape = a.shape
        self.a = a.flatten()
        self.z = z.flatten()
        self.n = self.a.size
        return self.a
    
    def backward_pass(self, dz):
        return dz.reshape(self.original_shape)

class Dense():
    def __init__(self, n=0, activation=None, initializer_w=glorot_uniform, initializer_b=zeros):
        self.n = n
        self.a = np.array([])
        if activation != None:
            self.w = np.array([])
            self.b = np.array([])
            self.z = np.array([])
            self.f = activation
            self.df = derivatives[activation]
            self.f_w = initializer_w
            self.f_b = initializer_b
    
class Conv():
    def __init__(self, branches=1, depth=1, size=1, stride=1, padding=0, activation=ReLU, initializer_w=glorot_uniform, initializer_b=zeros_map):
        self.branches = branches
        self.depth = depth
        self.size = size
        self.stride = stride
        self.padding = padding
        
        # functions
        self.f = activation
        self.df = derivatives.get(activation)
        self.f_b = initializer_b
        
        # kernels
        self.w = [[initializer_w(size, size) for _ in range(depth)] for _ in range(branches)]
        
        # other parameters
        self.b = np.array([])
        self.a = np.array([])
        self.z = np.array([])

    def forward_pass(self, a):
        size_x, size_y = a.shape
        
        choices_pad_x = size_x - self.size + 2 * self.padding
        choices_pad_y = size_y - self.size + 2 * self.padding

        x = choices_pad_x // self.stride + 1
        y = choices_pad_y // self.stride + 1

        self.b = [self.f_b(x, y) for _ in range(self.branches)]
        
        for kernel, b in zip(self.w, self.b):
            z = b
            for w in kernel:
                # add a padding box - up and down, left and right
                if self.padding:
                    a = np.pad(a, pad_width=self.padding, mode="constant")

                feature_map = []
                # do convolution between the array of data and the kernels' weights
                for i in range(0, choices_pad_y, self.stride):
                    row = []
                    for j in range(0, choices_pad_x, self.stride):
                        z_dash = a[i : i + self.size, j : j + self.size] * w
                        row.append(np.sum(z_dash))
                    feature_map.append(row)

                feature_map = np.array(feature_map, dtype=float)
                z += feature_map

            self.z = z
        try:
            self.a = self.f(self.z) # apply non-linearity
            return self.a
        except:
            raise ModuleNotFoundError("There is no non-linearity function given.")

class Pooling():
    def __init__(self, size=1, function="max"):
        self.original_shape = None
        self.size = size
        self.f = function
        self.a = np.array([])
        self.z = np.array([])
    
    def forward_pass(self, a, z=None):
        def pool_down(x):
            feature_map = x
            size_x, size_y = feature_map.shape
            
            # do pooling for the feature map
            if self.f == "global_max":
                a = np.array([np.max(feature_map)], dtype=float)
            elif self.f == "global_min":
                a = np.array([np.min(feature_map)], dtype=float)
            elif self.f == "global_avg":
                a = np.array([np.average(feature_map)], dtype=float)
            else:
                a = np.zeros((size_y - self.size + 1, size_x - self.size + 1))
                for i in range(0, size_y - self.size + 1):
                    for j in range(0, size_x - self.size + 1):
                        region = feature_map[i : i + self.size, j : j + self.size]
                
                        if self.f == "max":
                            a[i, j] = np.max(region)
                        elif self.f == "min": 
                            a[i, j] = np.min(region)
                        elif self.f == "avg": 
                            a[i, j] = np.average(region)  
                        else: 
                            a[i, j] = self.f(region)
            return a
        
        self.original_shape = a.shape
        
        if len(a) > 0:
            self.a = pool_down(a)
        else:
            raise ValueError("[a]'s size should not be zero.")
        if z != None:
            if z.shape != a.shape:
                raise ValueError("[z] should match the shape of [a].")
            self.z = pool_down(z)
        
        return self.a

    def backward_pass(self, dz):
        unpooled_map = np.zeros(self.original_shape, dtype=float)
        for i, y in enumerate(dz):
            for j, x in enumerate(y):
                expanded_x = np.full((self.size, self.size), fill_value=x, dtype=float)
                unpooled_map[i : i + self.size, j : j + self.size] += expanded_x 
        return unpooled_map

# Network
class nn():
    def __init__(self, *sequence, possible_outcomes=None, cost=None, dcost=None, optimizer=None):
        self.possible_outcomes = possible_outcomes
        self.optim = optimizer
        self.c, self.dc = cost, dcost

        # check for automated configuration
        self.stages = sequence
        for i, s in enumerate(sequence, 1):
            if isinstance(s, (Dense, Flatten)) and isinstance(sequence[i - 1], (Conv, Pooling)):
                self.stages.insert(i + 1, Flatten())
            elif isinstance(s, (Conv, Pooling)) and isinstance(sequence[i - 1], (Dense, Flatten)):
                self.stages.insert(i + 1, Reshape())
        self.n_stages = len(self.stages)
        self.layers = [s for s in sequence if isinstance(s, (Dense, Conv))]
        self.n_layers = len(self.layers)

        # inform the optimizer of the CNN's structure by
        # passing the number of stages to the optimizer
        if optimizer != None:
            optimizer.N = self.n_stages

    def feedforward(self, a):
        prev_shape = a.shape
        s = self.stages[0]

        if isinstance(s, (Flatten, Conv, Pooling)):
            a = s.forward_pass(a)
        
        elif isinstance(s, Dense):
            s.a = a.flatten()
        
        else: raise EnvironmentError("Your sturcture should start with ConvLayer, DenseLayer, FlattenLayer, ReshapeLayer, or Pooling.")
        
        for i, s in enumerate(self.stages[1:]):
            prev_s = self.stages[i]
            if isinstance(s, (Flatten, Pooling)):
                a = s.forward_pass(a, prev_s.z)

            elif isinstance(s, Reshape):
                if s.shape == -1:
                    size = 1
                    for x in prev_shape: size *= x
                    
                    ratio = a.size // size + 1
                    s.shape = prev_shape = (x * ratio for x in prev_shape)
                a = s.forward_pass(a, prev_s.z)
            
            elif isinstance(s, Conv):
                a = s.forward_pass(a)
            
            elif isinstance(s, Dense):
                # initialize weights and baises
                if len(s.w) == 0:
                    n_in = self.stages[i].n
                    n_out = s.n

                    # register these new pramaters
                    s.w, s.b = s.f_w(n_in, n_out), s.f_b(n_out)
                
                # calculate activations
                z = s.w @ a + s.b

                a = s.f(z)
                s.z = z
                s.a = a
            
            prev_shape = a.shape

        return a

    def loss(self, y):
        return self.c(self.output_vector(), y)

    def calculate_dz(self, i, dz):
        s = self.stages[i]
        if isinstance(s, (Dense, Conv)): 
            prev_s = self.stages[i - 1]
            df = self.layers[self.layers.index(s) - 1].df
            if isinstance(s, Conv):
                da = [[w.T @ dz for w in k] for k in s.w]
            else:
                da = s.w.T @ dz
            return da * df(prev_s.z)
        return s.backward_pass(dz)

    def backprop(self, y):
        # we will take the derivitative of the cost associated with 
        # a particular datapoint with respect to each neuron in 
        # the last layer. 
        
        dW, dB = [], []
        
        # initialize dz from the last layer
        if self.dc == None:
            try:
                dz = derivatives[self.c](self.stages[-1], y)
            except:
                raise ValueError(f"No differential expression is assigned to ({self.c.__name__})")
        else:
            dz = self.dc(self.stages[-1], y) * self.df(self.stages[-1].z)

        # now, we will work things backward
        for i in reversed(range(self.n_stages)):
            s = self.stages[i]
            if isinstance(s, (Dense, Conv)):
                # calculate the loss gradient with respect to 
                # the weights and the biases
                dw = np.outer(dz, self.stages[i - 1].a)
                db = dz

                if self.optim != None:
                    dw, db = self.optim.func(dw, db, i)

                dW.append(dw)
                dB.append(db)

            if i > 1 and len(self.stages[i - 1].z):
                dz = self.calculate_dz(i, dz)

        dW.reverse()
        dB.reverse()
        return dW, dB 

    def learn(self, x_data, y_data, targets, lr=0.01, epochs=1, batch_size=1):
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
                for i in reversed(range(self.n_layers)):
                    l = self.layers[i]
                    
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
            batch = data[i : batch_size + i]
            loss, n_correct = 0, 0
            for x, y, t in batch:
                self.feedforward(x)
                loss += self.loss(y)
                n_correct += int(self.output() == t)
            
            # calculate loss and accuarcy per batch
            try:
                acc_percentage = np.round(n_correct / len(batch) * 100, 2)
                loss_percentage = np.round(loss / len(batch) * 100, 2)
            except:
                raise ZeroDivisionError(f"(batch_size) should not be zero.")

            loss_traj.append(loss_percentage)
            accuarcy.append(acc_percentage)

            print(f"accuracy-{i + 1}:", f"{round(acc_percentage, 2)}%")
        return loss_traj, accuarcy
        
    def set_optimizer(self, optimizer=None):
        self.optim = optimizer

    def output_vector(self):
        return self.stages[-1].a
    
    def output(self):
        last_stage = self.stages[-1].a
        out_index = np.argmax(last_stage)
        return self.possible_outcomes[out_index]
    
# 1. do fast convolution
# 2. train kernel's weight, biases
# 3. add branching feature