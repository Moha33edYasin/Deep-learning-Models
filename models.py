import numpy as np
from methods import derivatives, ReLU, glorot_uniform, zeros, zeros_map 
from methods import align_and_pad, cross_corr, cross_corr_transposed, conv_transposed

# Store 1D | 2D | 3D tensor 
class Input():
    def __init__(self, a):
        self.previous = None
        self.a = a
        self.z = np.array([])
        self.df = None

# 3D | 2D | 1D tensor --> 3D | 2D | 1D tensor
class Reshape():
    def __init__(self, shape=-1):
        self.previous = None
        self.shape = shape
        self.original_shape = None
        self.df = None
        self.a = np.array([])
        self.z = np.array([])
    
    # 3D | 2D | 1D tensor --> 3D | 2D | 1D tensor
    def forward_pass(self, previous=None):
        self.previous = previous
        self.original_shape = previous.a.shape
        self.df = previous.df
        self.a = previous.a.reshape(self.shape)
        self.z = previous.z.reshape(self.shape)
        return self.a

    # 3D | 2D | 1D tensor <-- 3D | 2D | 1D tensor
    def backward_pass(self, dz_next):
        return dz_next.reshape(self.original_shape)

# 3D | 2D tensor --> 1D tensor
class Flatten():
    def __init__(self):
        self.previous = None
        self.original_shape = None
        self.df = None
        self.n = 0
        self.a = np.array([])
        self.z = np.array([])
    
    # 3D | 2D tensor --> 1D tensor
    def forward_pass(self, previous):
        self.previous = previous
        self.original_shape = previous.a.shape
        self.df = previous.df
        # self.a = np.hstack([a.flatten() for a in A])
        # self.z = np.hstack([z.flatten() for z in Z])
        self.a = previous.a.flatten()
        self.z = previous.z.flatten()
        self.n = self.a.size
        return self.a
    
    # 3D | 2D tensor <-- 1D tensor
    def backward_pass(self, dz_next):
        return dz_next.reshape(self.original_shape)

# 1D tensor --> 1D tensor
class Dense():
    def __init__(self, n=0, activation=None, initializer_w=glorot_uniform, initializer_b=zeros):
        self.n = n
        self.previous = None
        self.a = np.array([])
        self.w = np.array([])
        self.b = np.array([])
        self.z = np.array([])
        self.f = activation
        self.df = derivatives[activation]
        self.f_w = initializer_w
        self.f_b = initializer_b

    # 1D tensor --> 1D tensor
    def forward_pass(self, previous):
        self.previous = previous

        # initialize weights and baises
        if self.w.size == 0:
            # register these new pramaters
            self.w, self.b = self.f_w(self.previous.n, self.n), self.f_b(self.n)
        
        # calculate activations
        self.z = self.w @ previous.a + self.b
        self.a = self.f(self.z)
        return self.a

    def calculate_gradient(self, dz):
        # calculate the loss gradient with respect to 
        # the weights and the biases for dense layer
        dw = np.outer(dz, self.previous.a)
        db = dz
        return dw, db

    # 1D tensor <-- 1D tensor
    def backward_pass(self, dz_next):
        da = self.w.T @ dz_next
        return da * self.previous.df(self.previous.z)

# 3D tensor --> 3D tensor
class Conv():
    def __init__(self, depth=1, kernels=[], stride=1, padding=0, activation=ReLU, initializer_w=glorot_uniform, initializer_b=zeros_map):
        self.depth = depth
        self.kernels = kernels
        self.stride = stride
        self.pad = padding
        self.pad_width = padding

        # store the previous element to form a hierarchy 
        self.previous = None

        # functions
        self.f = activation
        self.df = derivatives.get(activation)
        self.f_b = initializer_b
        
        # kernels
        self.w = [[initializer_w(k[0], k[1]) for _ in range(depth)] for k in kernels]
        
        # other parameters
        self.b = np.array([])
        self.z = np.array([])
        self.a = np.array([])

    # 3D tensor --> 3D tensor
    def forward_pass(self, previous):
        self.previous = previous
        b = []
        z = []

        for k in self.w:
            height = self.previous.a[0].shape[0] + 2 * self.pad
            width = self.previous.a[0].shape[1] + 2 * self.pad
            
            # This corrects the input boundaries to align with kernel shape,
            # so that the kernel will cover all the input with the least padding 
            # room possible.
            (u, d), (l, r) = align_and_pad((height, width), k[0].shape, self.stride)
            self.pad_width = (u + self.pad, d + self.pad), (l + self.pad, r + self.pad)

            out = np.sum([cross_corr(x, w, self.stride, self.pad_width) for x, w in zip(previous.a, k)], axis=0)
            b_n = self.f_b(*out.shape)
            b.append(b_n)
            z.append(out + b_n)
        self.b = np.array(b, dtype=float)        
        self.z = np.array(z, dtype=float)        
        self.a = np.array(self.f(z), dtype=float)
        return self.a

    def calculate_gradient(self, dz):
        # calculate the loss gradient with respect to the
        # weights and the biases for convolutional layer
        dw = [[cross_corr_transposed(x, dz_n, k, self.stride, self.pad_width) for x in self.previous.a] for k, dz_n in zip(self.kernels, dz)]
        dw = np.array(dw, dtype=float)
        db = dz
        return dw, db

    # 3D tensor <-- 3D tensor
    def backward_pass(self, dz_next):
        dz = [[conv_transposed(k, dz_n, x.shape, self.stride, 'full') for k in kernel] for dz_n, x, kernel in zip(dz_next, self.previous.a, self.w)]
        dz = np.sum(dz, axis=0)
        return dz

# 3D tensor --> 3D tensor
class Pooling():
    def __init__(self, size=(1,1), function="max"):
        self.size = size
        self.previous = None
        
        # activations
        self.f = function
        self.df = None
        
        # parameters
        self.masks = np.array([]) # to distribute the gradient based on each input contribution 
        self.a = np.array([])
        self.z = np.array([])

    # 3D tensor --> 3D tensor
    def forward_pass(self, previous):
        self.previous = previous
        masks = []
        pooled_z = []
        pooled_a = []
        n_row, n_col = previous.a[0].shape[0] - self.size[0] + 1, previous.a[0].shape[1] - self.size[1] + 1
        if n_row < 0 or n_col < 0:
            raise SystemError(f"A diminished input with shape less than pooling window size is passed ({previous.a[0].shape} < {self.size})")
        if previous.z.any():
            for a_2d, z_2d in zip(previous.a, previous.z):
                # perform pooling to the feature map
                pooled_2d_z = np.zeros((n_row, n_col))
                pooled_2d_a = np.zeros((n_row, n_col))
                bin_mask = np.zeros(a_2d.shape, dtype=bool)
                for i in range(n_row):
                    for j in range(n_col):
                        z_region = z_2d[i : i + self.size[0], j : j + self.size[1]]
                        a_region = a_2d[i : i + self.size[0], j : j + self.size[1]]
                
                        if self.f == "max":
                            z_target = z_region.max()
                            a_target = a_region.max()
                            region_mask = np.where(a_region == a_target, True, False).astype(np.bool)
                        elif self.f == "min":
                            z_target = z_region.min()
                            a_target = a_region.min()
                            region_mask = np.where(a_region == a_target, True, False).astype(np.bool)
                        elif self.f == "avg":
                            z_target = np.average(z_region)
                            a_target = np.average(a_region)
                            region_mask = np.full(self.size, True, dtype=np.bool)

                        pooled_2d_z[i, j] = z_target
                        pooled_2d_a[i, j] = a_target
                        bin_mask[i : i + self.size[0], j : j + self.size[1]] += region_mask
                
                if self.f in ["max", "min"]:
                    mask = bin_mask.astype(int)
                elif self.f == "avg":
                    mask = np.where(bin_mask == True, 1 / self.size, 0).astype(float)

                masks.append(mask)
                pooled_z.append(pooled_2d_z)
                pooled_a.append(pooled_2d_a)

            self.z = np.array(pooled_z, dtype=float)
        else:
            for a_2d in previous.a:
                # perform pooling to the feature map
                pooled_2d_a = np.zeros((n_row, n_col))
                bin_mask = np.zeros(a_2d.shape, dtype=bool)
                for i in range(n_row):
                    for j in range(n_col):
                        a_region = a_2d[i : i + self.size[0], j : j + self.size[1]]
                
                        if self.f == "max":
                            a_target = a_region.max()
                            region_mask = np.where(a_region == a_target, True, False).astype(np.bool)
                        elif self.f == "min":
                            a_target = a_region.min()
                            region_mask = np.where(a_region == a_target, True, False).astype(np.bool)
                        elif self.f == "avg":
                            a_target = np.average(a_region)
                            region_mask = np.full(self.size, True, dtype=np.bool)

                        pooled_2d_a[i, j] = a_target
                        bin_mask[i : i + self.size[0], j : j + self.size[1]] += region_mask
                
                if self.f in ["max", "min"]:
                    mask = bin_mask.astype(int)
                elif self.f == "avg":
                    mask = np.where(bin_mask == True, 1 / self.size, 0)

                masks.append(mask)
                pooled_a.append(pooled_2d_a)

        self.df = previous.df
        self.masks = masks
        self.a = np.array(pooled_a, dtype=float)
        return self.a

    # 3D tensor <-- 3D tensor
    def backward_pass(self, dz_next):
        result = []
        for mask, dz_2d in zip(self.masks, dz_next):
            n_row, n_col = mask.shape[0] - self.size[0] + 1, mask.shape[1] - self.size[1] + 1
            distributed_dz = np.zeros(mask.shape)
            for i in range(n_row):
                for j in range(n_col):
                    distributed_dz[i : i + self.size[0], j : j + self.size[1]] += mask[i : i + self.size[0], j : j + self.size[1]] * dz_2d[i, j]
            result.append(distributed_dz)
        result = np.array(result, dtype=float)
        return result

# 3D tensor --> 1D tensor
class Global_Pooling():
    def __init__(self, function="max"):        
        self.previous = None
        
        # activations
        self.f = function
        self.df = None

        # parameters
        self.masks = [] # to distribute the gradient based on each input contribution 
        self.a = []
        self.z = []

    # 3D tensor --> 1D tensor
    def forward_pass(self, previous):
        self.previous = previous
        self.df = previous.df
        masks = []
        pooled_z = []
        pooled_a = []
        
        for a_2d, z_2d in zip(previous.a, previous.z):
            # perform pooling to the feature map
            if self.f == "max":
                z_target = z_2d.max()
                a_target = a_2d.max()
                mask = np.where(a_2d == a_2d.max()).astype(int)
            elif self.f == "min":
                z_target = z_2d.min()
                a_target = a_2d.min()
                mask = np.where(a_2d == a_2d.min()).astype(int)
            elif self.f == "avg":
                z_target = np.average(z_2d)
                a_target = np.average(a_2d)
                mask = np.full(a_2d.shape, 1 / a_2d.size, dtype=float)

            masks.append(mask)
            pooled_z.append(z_target)
            pooled_a.append(a_target)

        self.masks = masks
        self.z = np.array(pooled_z, dtype=float)
        self.a = np.array(pooled_a, dtype=float)
        return self.a

    # 3D tensor <-- 1D tensor
    def backward_pass(self, dz_next):
        result = []
        for mask, dz_2d in zip(self.masks, dz_next):
            distributed_dz = mask * dz_2d
            result.append(distributed_dz)
        return np.array(result, dtype=float)

class Adaptive_Pooling(Pooling):
    def __init__(self, output_size=(1, 1), function="max"):
        self.output_size = output_size
        super().__init__((1, 1), function)

    def forward_pass(self, A, Z=np.array([0]), df=None):
        self.size = (A.shape[0] - self.output_size[0] + 1, A.shape[1] - self.output_size[1] + 1)
        return super().forward_pass(A, Z, df)

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
        self.layers = [s for s in sequence if isinstance(s, (Dense, Conv))]

        # inform the optimizer of the CNN's structure by
        # passing the number of stages to the optimizer
        if optimizer != None:
            optimizer.N = len(self.stages)

    def feedforward(self, a):
        s = self.stages[0]

        if isinstance(s, Flatten):
            s.a = a = np.array(a, dtype=float).flatten()
            s.n = len(s.a)
        elif isinstance(s, Reshape):
            s.a = a = np.array(a, dtype=float).reshape(s.shape)
            s.n = len(s.a)
        else:
            # Input(a) works like starting point to the nn if a flatten 
            # or a reshape layer isn't in that starting point 
            a = s.forward_pass(Input(a))
            
        for i, s in enumerate(self.stages[1:]):
            prev_s = self.stages[i]

            if isinstance(s, Reshape):
                if s.shape == -1:
                    next_stage = self.stages[i + 2]
                    n_branch = next_stage.branches
                    d1 = round(a.size / n_branch)
                    d2 = round(d1 / next_stage.shape[0])
                    s.shape = (d1, d2, next_stage.shape[0])
            a = s.forward_pass(prev_s)
        return a

    def loss(self, y):
        return self.c(self.output_vector(), y)

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
        for i in reversed(range(len(self.stages))):
            s = self.stages[i]
            if s in self.layers:
                dw, db = s.calculate_gradient(dz)
                if self.optim != None:
                    dw, db = self.optim.func(dw, db, i)
                
                dW.append(dw)
                dB.append(db)

            if s.previous.previous:
                if len(s.previous.z):
                    dz = s.backward_pass(dz)

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
                for i in reversed(range(len(self.layers))):
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

  
# TODO: train kernel's weight, biases
# ! implement fast convolution
# * add branching feature
