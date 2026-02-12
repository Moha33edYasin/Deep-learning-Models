# What is it?
A neural network framework implemented entirely from scratch, including a multi-layer perceptron (MLP) and convolutional neural network (CNN). The project recreates core deep learning components such as kernels, pooling layers, dense/flatten transitions, channel handling, optimizers, activation functions, and parameter initialization strategies. The framework emphasizes modularity, transparency, and user control.  

# Why? 
To develop a first-principles understanding of neural network mechanics rather than relying on high-level libraries.  

# Next step:  
Extending the system toward an intelligent agent capable of interpreting human voice commands to drive actions in a 2D environment.  

## Usage

Import the module.  
```python
from deep_learning.models import *
from deep_learning.activitions import *
```

To use NN model, call `NeuralNetwork()` and configure your structure.  
*Example:*  

```python
nn = NeuralNetwork(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                   Layer(784),
                   Layer(16, RelU),
                   Layer(16, RelU),
                   Layer(10, softmax),
                   cost= CCE, # CCE means Categorical Cross-Entropy
                   optimizer= Adam(β1=0.9, β2=0.99, lr=0.05)
                   )
```

To train the model, prepare an input-field set of data and its output field, and call `learn()`, which return the the **loss** and **accuracy** across all **epochs** of the training process.  
*Example:*   

```python
# Note: in this case `xtrain` and `ytrain` are the input and expected output fields
loss1, acc1 = nn.learn(xtrain, ytrain, ttrain, lr=0.01, epochs=epochs, batch_size=batch_size)
```

To test the model, prepare an input-field set of data and its output field other than those used in training, and call `test()`, which return the the **loss** and **accuracy** across all **batches** of the testing dataset.  
*Example:*  

```python
# Note: in this case `xtest` and `ytest` are the input and expected output fields
loss2, acc2 = nn.test(xtest, ytest, ttest, batch_size)
```
