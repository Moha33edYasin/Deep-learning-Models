# What is it?
A *neural network framework* implemented entirely from scratch, featuring both a *Multi-Layer Perceptron (MLP)* and a *Convolutional Neural Network (CNN)*.  

The project reconstructs fundamental deep learning components, including convolution kernels, pooling operations, dense-to-flatten transitions, channel handling, optimizers, activation functions, and parameter initialization strategies. The framework is designed with an emphasis on modularity, transparency, and full user control, enabling direct interaction with the underlying learning mechanics.  

# Why? 
To develop a first-principles understanding of neural network mechanics rather than relying on high-level libraries.  
Re-implementing core mechanisms provided deeper insight into how learning dynamics, gradient flow, and architectural design choices interact.  

# Next step:  
Extending the system toward an intelligent agent capable of interpreting human voice commands to mapping them into actions within a 2D environment.
> [!Declaration]  
> Contribution will be appreciated.  

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
                   Layer(16, ReLU),
                   Layer(16, ReLU),
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
