# What is it?

A **neural network framework** implemented entirely from scratch. It reconstructs fundamental deep learning components, including _convolution kernels, pooling operations, auto-selecting for the optimal convolution algorithm, channel handling, optimizers, activation functions, and parameter initialization strategies_.  

The framework emphasizes **modularity, transparency, and user control**, enabling direct inspection and control of the learning mechanics.  

# Why?

To develop a first-principles understanding of neural network behavior rather than relying on high-level libraries.  
Re-implementing core mechanisms provides deeper insight into learning dynamics, gradient flow, and architectural design trade-offs.  

# Setup

**Requirements**:  
- C++20  
- CMake  
- pybind11
- Eigen 5.0.0
- numpy
- matplotlib

Then, type:  
```bash
cmake -S src -B build
cmake --build build
```
This will create the necessary `.pyd` file.  

After that, run:  
```
python -m __init__
```
To check that everything is working.  

# Usage

Import the modules:  

```python
from netjet.models import *
from netjet.methods import *
```  

Instantiate a neural network and define the architecture:  

```python
mlp = nn(
        Flatten(),
        Dense(16, ReLU),
        Dense(16, ReLU),
        Dense(10, softmax)
)
```

compile the model:  
```python
mlp.compile(
        input_shape=(128, 1, 28, 28), # (batch_size, channels, height, width)
        possible_outcomes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        cost= CCE, # Cross Categorical Entropy
        optimizer= Adam(lr=0.001)
)
```
you would see something like this:
```bash
(25%) : fused Flatten (0) --> Input Node.
(50%) : fused Dense (1) --> Flatten (0).
...
(100%) : fused ...
Adam optimization is configured.
(*) The network is compiled successfully.       (0.01s)
```
Train the model:  
```python
loss1, acc1 = mlp.learn(xtrain, ytrain, ttrain, epochs=8)
```
```bash
(#) accuracy per epoch (1): 78.66%       (2.03s)
(#) accuracy per epoch (2): 92.11%       (2.07s)
...
(#) accuracy per epoch (...): ...%       (~2.0s)
(*) Training is complete.       (16.53s)
```

Also you can inspect your training by setting `Debug_plot=True` in `learn`:  
<img width="930" height="376" alt="Python 3 11 6_9_2026 2_20_54 PM" src="https://github.com/user-attachments/assets/77c9f3e5-2719-4b5d-8f3b-de65018fa011" />

> [!NOTE]
> Enabling `Debug_plot` can increase learning time, so use it wisely to refine your model.

Evaluate on test data:  
```python
loss2, acc2 = mlp.test(xtest, ytest, ttest)
```
> [!NOTE]
> * The above structure achieved `~94-95%` testing accuarcy on MNIST.  
> * `Dense`, `Flatten`, and `Reshape` layers are relatively fast.  

To use convolutional layers:  
```python
cnn = nn(
            Conv(k_shape=(3, 3), n_kernels=2, activation=ReLU),
            MaxPool((2, 2)),
            Conv(k_shape=(3, 3), n_kernels=4, activation=ReLU),
            MaxPool((2, 2)),
            Flatten(),
            Dense(16, ReLU),
            Dense(10, softmax)
)

```  
> [!NOTE]
> * With `8 epochs`, a `batch size` of `128` and `Adam`, The above structure achieved `~96-97%` testing accuarcy on MNIST (~15s/epoch).  
> * Convolution and pooling operations are now moderately faster.  

**Use `set` to change the flow of the input and output (or more concretely changing modes).**  
`0` is default mode, aka. forward-and-backward-pass network ("backward"/"trainable" can also be used).  
`1` is single-forward pass. very intuitive as it set `batch_size=1` ("single-forward" can also be used).    
`2` is  forward-only network. Backpropagtion is not possible  ("forward"/"frozen" can also be used).  
This can be done like this:  

```python
cnn.set(0) # or cnn.set("backward") / cnn.set("backward")
cnn.set(2) # or cnn.set("forward") / cnn.set("frozen")
```


---
# Happy with the result? Save it
To save use `save` with the `nn` class.
for the above model, we save it and named it "cnn_for_mnist":
```python
cnn.save("cnn_for_mnist")
```

To get the saving back again or load it into other model, use `fetch`.
for our model above: 
```python
cnn.fetch("cnn_for_mnist") # recover back
```
You can delete your saving, using `delete` from `methods` file like this:  
```python
from methods import delete
delete("cnn_for_mnist")
```
Or you may delete all saving at once using `clean_local_storage()` from `methods` file.

> [!NOTE]
> If there was no `local_storage` folder, any of the above operations will create that folder the second time you touch one of them in your code.

You can also use `copy` to replicate model architecture, and `copy_from` to pull model parameters into other one.
for our two models above:
```python
mlp.copy(cnn)       # mlp contain the same sequence of layers and operations as cnn 
mlp.copy_from(cnn)  # mlp == cnn (same parameters, same architecture)
```

---
# Built-in Layers
- Dense Layers: `Dense(number of neuron, activation, weight_initailization, biases_initailization)`  
- Convolutional Layers: `Conv(number of kernels, kernel shape, padding, stride, activation, weight_initailization, biases_initailization)`  
- Flatten Layer: `Flatten()`  
- Reshape Layer: `Reshape(2d shape)`  
- Max pooling: `Maxpool(2d shape, stride)`  
- Min pooling: `Minpool(2d shape, stride)`  
- Average pooling: `Averagepool(2d shape, stride)`  
- Global max pooling: `GlobalMaxPool()`  
- Global min pooling: `GlobalMinPool()`  
- Global average pooling: `GlobalAveragePool()`  
- Adaptive max pooling: `AdaptiveMaxPool(2d out_shape, stride)`  
- Adaptive min pooling: `AdaptiveMinPool(2d out_shape, stride)`  
- Adaptive average pooling: `AdaptiveAveragePool(2d out_shape, stride)`

> [!IMPORTANT]
> **Convolution Naming & Gradient Implementation:**  
> The names I use for convolution operations in `src/` do not necessarily follow standard terminology.   
> Additionally, the gradient calculations are derived and implemented manually not following a standard framework implementation. As a result, some of the methods or terminology may appear unconventional or out of place.  

# Built-in initialization techniques
- He initialization :`he_normal`, `he_uniform`  
- Glorot initialization: `glorot_normal`, `glorot_uniform`  
- Zero initialization: `zeros`
 
# Built-in activation functions
- `sigmoid`, `softmax`, `ReLU`, `Leaky_ReLU`

# Built-in Cost functions
- `MSE` : Mean Squared Error  
- `BCE` : Binary Cross Entropy  
- `CCE`: Cross Categorical Entropy
- `Huber(δ)`: Huber, with `δ` being threshold parameter   

# Built-in Optimizers
- `SGD`, `Momentem`, `Nestrov_A`, `AdaGrad`, `AdaDelta`, `RMSProp`, `AdaMax`, `Adam`, `nAdam`, and `AMSGrad`  

---
For more examples, you may experiment with `mnist_test.py` in `examples` and run it using:  
```bash
python examples/mnist_test
```
