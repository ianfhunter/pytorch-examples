This repository introduces the fundamental concepts of
[PyTorch](https://github.com/pytorch/pytorch)
through self-contained examples.

At its core, PyTorch provides two main features:
- An n-dimensional Tensor, similar to numpy but can run on GPUs
- Automatic differentiation for building and training neural networks

We will use a fully-connected ReLU network as our running example. The network
will have a single hidden layer, and will be trained with gradient descent to
fit random data by minimizing the Euclidean distance between the network output
and the true output.

### Table of Contents

- <a href='#tensors'>Tensors</a>
- <a href='#variables-and-autograd'>Variables and autograd</a>
- <a href='#nn-the-neural-network-library'>nn: The neural network Library</a>
- <a href='#optim-the-optimization-library'>optim: The optimization library</a>
- <a href='#rnns'>RNNs</a>
- <a href='#data-loading'>Data Loading</a>
- <a href='#pytorch-for-torch-users'>PyTorch for Torch Users</a>
- <a href='#advanced-topics'>Advanced Topics </a>

## Tensors

Numpy is a great framework, but it cannot utilize GPUs to accelerate its
numerical computations. For modern deep neural networks, GPUs often provide
speedups of [50x or greater](https://github.com/jcjohnson/cnn-benchmarks), so
unfortunately numpy won't be enough for modern deep learning.

Here we introduce the most fundamental PyTorch concept: the **Tensor**. A PyTorch
Tensor is conceptually identical to a numpy array: a Tensor is an n-dimensional
array, and PyTorch provides many functions for operating on these Tensors. Like
numpy arrays, PyTorch Tensors do not know anything about deep learning or
computational graphs or gradients; they are a generic tool for scientific
computing.

However unlike numpy, PyTorch Tensors can utilize GPUs to accelerate their
numeric computations. To run a PyTorch Tensor on GPU, you simply need to cast it
to a new datatype.

Here we use PyTorch Tensors to fit a two-layer network to random data. Like the
numpy example above we need to manually implement the forward and backward
passes through the network:

```python
# Code in file tensor/two_layer_net_tensor.py
import torch

dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = torch.randn(N, D_in).type(dtype)
y = torch.randn(N, D_out).type(dtype)

# Randomly initialize weights
w1 = torch.randn(D_in, H).type(dtype)
w2 = torch.randn(H, D_out).type(dtype)

learning_rate = 1e-6
for t in range(500):
  # Forward pass: compute predicted y
  h = x.mm(w1)
  h_relu = h.clamp(min=0)
  y_pred = h_relu.mm(w2)

  # Compute and print loss
  loss = (y_pred - y).pow(2).sum()
  print(t, loss)

  # Backprop to compute gradients of w1 and w2 with respect to loss
  grad_y_pred = 2.0 * (y_pred - y)
  grad_w2 = h_relu.t().mm(grad_y_pred)
  grad_h_relu = grad_y_pred.mm(w2.t())
  grad_h = grad_h_relu.clone()
  grad_h[h < 0] = 0
  grad_w1 = x.t().mm(grad_h)

  # Update weights using gradient descent
  w1 -= learning_rate * grad_w1
  w2 -= learning_rate * grad_w2
```

## Variables and autograd

In the above examples, we had to manually implement both the forward and
backward passes of our neural network. Manually implementing the backward pass
is not a big deal for a small two-layer network, but can quickly get very hairy
for large complex networks.

Thankfully, we can use
[automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)
to automate the computation of backward passes in neural networks. 
The **autograd** package in PyTorch provides exactly this functionality.
When using autograd, the forward pass of your network will define a
**computational graph**; nodes in the graph will be Tensors, and edges will be
functions that produce output Tensors from input Tensors. Backpropagating through
this graph then allows you to easily compute gradients.

This sounds complicated, it's pretty simple to use in practice. We wrap our
PyTorch Tensors in **Variable** objects; a Variable represents a node in a
computational graph. If `x` is a Variable then `x.data` is a Tensor, and
`x.grad` is another Variable holding the gradient of `x` with respect to some
scalar value.

PyTorch Variables have the same API as PyTorch Tensors: (almost) any operation
that you can perform on a Tensor also works on Variables; the difference is that
using Variables defines a computational graph, allowing you to automatically
compute gradients.

Here we use PyTorch Variables and autograd to implement our two-layer network;
now we no longer need to manually implement the backward pass through the
network:

```python
# Code in file autograd/two_layer_net_autograd.py
import torch
from torch.autograd import Variable

dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs, and wrap them in Variables.
# Setting requires_grad=False indicates that we do not need to compute gradients
# with respect to these Variables during the backward pass.
x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

# Create random Tensors for weights, and wrap them in Variables.
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Variables during the backward pass.
w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

learning_rate = 1e-6
for t in range(500):
  # Forward pass: compute predicted y using operations on Variables; these
  # are exactly the same operations we used to compute the forward pass using
  # Tensors, but we do not need to keep references to intermediate values since
  # we are not implementing the backward pass by hand.
  y_pred = x.mm(w1).clamp(min=0).mm(w2)
  
  # Compute and print loss using operations on Variables.
  # Now loss is a Variable of shape (1,) and loss.data is a Tensor of shape
  # (1,); loss.data[0] is a scalar value holding the loss.
  loss = (y_pred - y).pow(2).sum()
  print(t, loss.data[0])
  
  # Manually zero the gradients before running the backward pass
  w1.grad.data.zero_()
  w2.grad.data.zero_()

  # Use autograd to compute the backward pass. This call will compute the
  # gradient of loss with respect to all Variables with requires_grad=True.
  # After this call w1.data and w2.data will be Variables holding the gradient
  # of the loss with respect to w1 and w2 respectively.
  loss.backward()

  # Update weights using gradient descent; w1.data and w2.data are Tensors,
  # w1.grad and w2.grad are Variables and w1.grad.data and w2.grad.data are
  # Tensors.
  w1.data -= learning_rate * w1.grad.data
  w2.data -= learning_rate * w2.grad.data
```

## nn: The neural network Library
Computational graphs and autograd are a very powerful paradigm for defining
complex operators and automatically taking derivatives; however for large
neural networks raw autograd can be a bit too low-level.

When building neural networks we frequently think of arranging the computation
into **modules**, some of which have **learnable parameters** which will be
optimized during learning.

In PyTorch, the `nn` package defines a set of
**Modules**, which are roughly equivalent to neural network layers. A Module receives
input Variables and computes output Variables, but may also hold internal state such as
Variables containing learnable parameters. The `nn` package also defines a set of useful
loss functions that are commonly used when training neural networks.

The **Module** class is also used to assemble layers into larger structures. Here we
define our two-layer network as a **Module** containing several submodules.
**Module** must only implement the `forward` method; the backwards computation is
handled automatically by autograd.

```python
# Code in file nn/two_layer_net_nn.py
import torch
import torch.nn as nn
from torch.autograd import Variable

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

class TwoLayerNet(nn.Module):
  def __init__(self, D_in, H, D_out):
    """
    In the constructor we instantiate two nn.Linear modules and assign them as
    member variables.
    """
    super(TwoLayerNet, self).__init__()
    self.linear1 = nn.Linear(D_in, H)
    self.linear2 = nn.Linear(H, D_out)

  def forward(self, x):
    """
    In the forward function we accept a Variable of input data and we must return
    a Variable of output data. We can use Modules defined in the constructor as
    well as arbitrary operators on Variables.
    """
    h_relu = self.linear1(x).clamp(min=0)
    y_pred = self.linear2(h_relu)
    return y_pred

model = TwoLayerNet(D_in, H, D_out)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = nn.MSELoss(size_average=False)

learning_rate = 1e-4
for t in range(500):
  # Forward pass: compute predicted y by passing x to the model. Module objects
  # override the __call__ operator so you can call them like functions. When
  # doing so you pass a Variable of input data to the Module and it produces
  # a Variable of output data.
  y_pred = model(x)

  # Compute and print loss. We pass Variables containing the predicted and true
  # values of y, and the loss function returns a Variable containing the loss.
  loss = loss_fn(y_pred, y)
  print(t, loss.data[0])
  
  # Zero the gradients before running the backward pass.
  model.zero_grad()

  # Backward pass: compute gradient of the loss with respect to all the learnable
  # parameters of the model. Internally, the parameters of each Module are stored
  # in Variables with requires_grad=True, so this call will compute gradients for
  # all learnable parameters in the model.
  loss.backward()

  # Update the weights using gradient descent. Each parameter is a Variable, so
  # we can access its data and gradients like we did before.
  for param in model.parameters():
    param.data -= learning_rate * param.grad.data
```

## optim: The optimization library
Up to this point we have updated the weights of our models by manually mutating the
`.data` member for Variables holding learnable parameters. This is not a huge burden
for simple optimization algorithms like stochastic gradient descent, but in practice
we often train neural networks using more sophisiticated optimizers like AdaGrad,
RMSProp, Adam, etc.

The `optim` package in PyTorch abstracts the idea of an optimization algorithm and
provides implementations of commonly used optimization algorithms.

In this example we will use the `nn` package to define our model as before, but we
will optimize the model using the Adam algorithm provided by the `optim` package:

```python
# Code in file nn/two_layer_net_optim.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

class TwoLayerNet(nn.Module):
  def __init__(self, D_in, H, D_out):
    super(TwoLayerNet, self).__init__()
    self.linear1 = nn.Linear(D_in, H)
    self.linear2 = nn.Linear(H, D_out)

  def forward(self, x):
    h_relu = self.linear1(x).clamp(min=0)
    y_pred = self.linear2(h_relu)
    return y_pred


model = TwoLayerNet(D_in, H, D_out)
loss_fn = nn.MSELoss(size_average=False)

# The optim package contains a number of data loaders, such as
# SGD, Adagrad, Adadelta, and Adam. You can optimize different parts
# of the model with different optimizers by passing a subset of
# model.parameters() to each optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)
for t in range(500):
  y_pred = model(x)

  loss = loss_fn(y_pred, y)
  print(t, loss.data[0])
  
  model.zero_grad()

  loss.backward()

  # Performs one Adam step on the parameters.
  # Since the gradients are stored inside the parameter Variables,
  # `step()` requires no arguments.
  optimizer.step()
```

## RNNs

RNNs are particularly easy to write in PyTorch because of its dynamic
graphs and imperative style; for example, here is a complete implementation of
a simple Ellman RNN.

```python
import torch.nn as nn
import torch.nn.functional as F
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RNN, self).__init__()
        self.ih = nn.Linear(input_dim, hidden_dim)
        self.hh = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input, hidden):
        """
        input: [seq_len x batch x input_dim] tensor
        hidden: [batch x hidden_dim] tensor
        """
        output = []
        for input_i in input:
            hidden = F.relu(self.ih(input_i) + self.hh(hidden))
            outputs.append(hidden)

        # joins the list of 2D tensors into a single 3D tensor
        output = torch.stack(output)

        return output, hidden
``` 

The `torch.nn.rnn` package contains building blocks for RNNs, GRUs, and LSTMs. 
These RNN modules have CuDNN support, but can also be run interchangeably without CuDNN
(e.g. on CPU).


## Data Loading

We often want to load inputs and targets from files, instead of using random inputs. We also often want to do any preprocessing in the background to avoid slowing down the training loop. PyTorch provides two classes `torch.utils.data.Dataset` and `torch.utils.data.DataLoader` to help with data loading. `DataLoader` implements batching and shuffling. It will load the data in background processes if you set `num_workers`.

```python
# Code in file nn/data_loading.py
import torch
import os
import PIL
from torch.utils.data import Dataset, DataLoader

class MyImageDataset(Dataset):
  def __init__(self, path):
    self.filenames = os.listdir(path)

  def __getitem__(self, i):
    pic = Image.open(self.filenames[i])
    # convert to RGB tensor
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    img = img.view(pic.size[1], pic.size[0], len(pic.mode))
    img = img.transpose(0, 1).transpose(0, 2).float() / 255
    return img

  def __len__(self):
    return len(self.filenames)

dataset = MyImageDataset('./images')

# load the images one at a time
for img in dataset:
  print(img.size())  # e.g. torch.Size(3, 28, 28)

# load, batch, and shuffle images using multiple workers
loader = DataLoader(dataset, num_workers=4, batch_size=64, shuffle=True)
for batch in loader:
  print(batch.size())  #e.g. torch.Size(64, 3, 28, 28)
```

PyTorch also provides a number of implementations for common datasets in the `vision` and `text` packages:
- Vision: MNIST, LSUN, COCO, CIFAR, and generic "ImageFolder"
- Text: SNLI, SST, and generic "Translation" and "LanguageModeling"

## PyTorch for Torch Users

The non-autograd parts of pytorch will be quite familiar to torch users, but there are
a few important changes to be aware of:

**Inplace / Out-of-place**

The first difference is that ALL operations on the tensor that operate in-place on it will have an **_** postfix.
For example, `add` is the out-of-place version, and `add_` is the in-place version.

```python
a.fill_(3.5)
# a has now been filled with the value 3.5

b = a.add(4.0)
# a is still filled with 3.5
# new tensor b is returned with values 3.5 + 4.0 = 7.5
```

Some operations like narrow do not have in-place versions, and hence, `.narrow_` does not exist. 
Similarly, some operations like `fill_` do not have an out-of-place version, so `.fill` does not exist.

 **Zero Indexing**

Another difference is that Tensors are zero-indexed. (Torch tensors are one-indexed)

```python
b = a[0,3] # select 1st row, 4th column from a
```

Tensors can be also indexed with Python's slicing

```python
b = a[:,3:5] # selects all rows, columns 3 to 5
```

**No camel casing**

The next small difference is that all functions are now NOT camelCase anymore.
For example `indexAdd` is now called `index_add_`

```python
x = torch.ones(5, 5)
print(x)
z = torch.Tensor(5, 2)
z[:,0] = 10
z[:,1] = 100
print(z)
x.index_add_(1, torch.LongTensor([4,0]), z)
print(x)
```

**Numpy Bridge**

Converting a torch Tensor to a numpy array and vice versa is a breeze.
The torch Tensor and numpy array will share their underlying memory, and changing one will change the other.

*Converting torch Tensor to numpy Array*

```python
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
a.add_(1)
print(a)
print(b) # see how the numpy array changed in value
```

*Converting numpy Array to torch Tensor*

```python
import numpy as np
a = np.ones(5)
b = torch.DoubleTensor(a)
np.add(a, 1, out=a)
print(a)
print(b) # see how changing the np array changed the torch Tensor automatically
```

All the Tensors on the CPU except a CharTensor support converting to NumPy and back.

**CUDA Tensors**

CUDA Tensors are nice and easy in pytorch, and they are much more consistent as well.
Transfering a CUDA tensor from the CPU to GPU will retain it's type.

```python
# creates a LongTensor and transfers it 
# to GPU as torch.cuda.LongTensor
a = torch.LongTensor(10).fill_(3).cuda()
print(type(a))
b = a.cpu()
# transfers it to CPU, back to 
# being a torch.LongTensor
```

**CUDA Caching Allocator**

Torch used the standard CUDA allocator, which meant that constructing and deleting
tensors caused GPU synchronization which was very slow. This led to a design pattern
of constructing all CUDA tensors once and only resizing them between iterations.

A caching allocator was developed for PyTorch (which is now in Torch as well!),
which removes the overhead for allocating and freeing cuda tensors. So it's now
fine to construct tensors within your training loop, e.g.

```python
for batch in batches:
  batch_gpu = batch.cuda()  # creates a new cuda tensor
  out = model(batch_gpu)
  ...
```

**Multiprocessing vs multithreading**

In Lua, CPU parallelism for data loading and HOGWILD typically used multithreading
via the torch `threads` package. In Python, CPU parallelism is achieved through the
torch `multiprocessing` package. This is a simple extension of the Python `multiprocessing`
package, that causes tensor storages to be passed between processes in shared memory.

Unlike the torch threads library, arbitrary objects (e.g. whole models) can be shared
between Python processes for HOGWILD training.

The [MNIST HOGWILD example](https://github.com/pytorch/examples/blob/master/mnist_hogwild/main.py) and the
[PyTorch data loader](https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataloader.py)
are good examples of how to use torch multiprocessing.

## Advanced Topics 

### Defining new autograd functions
Under the hood, each primitive autograd operator is really two functions that
operate on Tensors. The **forward** function computes output Tensors from input
Tensors. The **backward** function receives the gradient of the output Tensors
with respect to some scalar value, and computes the gradient of the input Tensors
with respect to that same scalar value.

In PyTorch we can easily define our own autograd operator by defining a subclass
of `torch.autograd.Function` and implementing the `forward` and `backward` functions.
We can then use our new autograd operator by constructing an instance and calling it
like a function, passing Variables containing input data.

In this example we define our own custom autograd function for performing the ReLU
nonlinearity, and use it to implement our two-layer network:

```python
# Code in file autograd/two_layer_net_custom_function.py
import torch
from torch.autograd import Variable

class MyReLU(torch.autograd.Function):
  """
  We can implement our own custom autograd Functions by subclassing
  torch.autograd.Function and implementing the forward and backward passes
  which operate on Tensors.
  """
  def forward(self, input):
    """
    In the forward pass we receive a Tensor containing the input and return a
    Tensor containing the output. You can cache arbitrary Tensors for use in the
    backward pass using the save_for_backward method.
    """
    self.save_for_backward(input)
    return input.clamp(min=0)

  def backward(self, grad_output):
    """
    In the backward pass we receive a Tensor containing the gradient of the loss
    with respect to the output, and we need to compute the gradient of the loss
    with respect to the input.
    """
    input, = self.saved_tensors
    grad_input = grad_output.clone()
    grad_input[input < 0] = 0
    return grad_input


dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs, and wrap them in Variables.
x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

# Create random Tensors for weights, and wrap them in Variables.
w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

learning_rate = 1e-6
for t in range(500):
  # Construct an instance of our MyReLU class to use in our network
  relu = MyReLU()
  
  # Forward pass: compute predicted y using operations on Variables; we compute
  # ReLU using our custom autograd operation.
  y_pred = relu(x.mm(w1)).mm(w2)
  
  # Compute and print loss
  loss = (y_pred - y).pow(2).sum()
  print(t, loss.data[0])
  
  # Manually zero the gradients before running the backward pass
  w1.grad.data.zero_()
  w2.grad.data.zero_()

  # Use autograd to compute the backward pass.
  loss.backward()

  # Update weights using gradient descent
  w1.data -= learning_rate * w1.grad.data
  w2.data -= learning_rate * w2.grad.data
```


### PyTorch vs TensorFlow: Static vs Dynamic Graphs
PyTorch autograd looks a lot like TensorFlow: in both frameworks we define
a computational graph, and use automatic differentiation to compute gradients.
The biggest difference between the two is that TensorFlow's computational graphs
are **static** and PyTorch uses **dynamic** computational graphs.

In TensorFlow, we define the computational graph once and then execute the same
graph over and over again, possibly feeding different input data to the graph.
In PyTorch, each forward pass defines a new computational graph.

Static graphs are nice because you can optimize the graph up front; for example
a framework might decide to fuse some graph operations for efficiency, or to
come up with a strategy for distributing the graph across many GPUs or many
machines. If you are reusing the same graph over and over, then this potentially
costly up-front optimization can be amortized as the same graph is rerun over
and over.

One aspect where static and dynamic graphs differ is control flow. For some models
we may wish to perform different computation for each data point; for example a
recurrent network might be unrolled for different numbers of time steps for each
data point; this unrolling can be implemented as a loop. With a static graph the
loop construct needs to be a part of the graph; for this reason TensorFlow
provides operators such as `tf.scan` for embedding loops into the graph. With
dynamic graphs the situation is simpler: since we build graphs on-the-fly for
each example, we can use normal imperative flow control to perform computation
that differs for each input.

To contrast with the PyTorch autograd example above, here we use TensorFlow to
fit a simple two-layer net:

```python
# Code in file autograd/tf_two_layer_net.py
import tensorflow as tf
import numpy as np

# First we set up the computational graph:

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create placeholders for the input and target data; these will be filled
# with real data when we execute the graph.
x = tf.placeholder(tf.float32, shape=(None, D_in))
y = tf.placeholder(tf.float32, shape=(None, D_out))

# Create Variables for the weights and initialize them with random data.
# A TensorFlow Variable persists its value across executions of the graph.
w1 = tf.Variable(tf.random_normal((D_in, H)))
w2 = tf.Variable(tf.random_normal((H, D_out)))

# Forward pass: Compute the predicted y using operations on TensorFlow Tensors.
# Note that this code does not actually perform any numeric operations; it
# merely sets up the computational graph that we will later execute.
h = tf.matmul(x, w1)
h_relu = tf.maximum(h, tf.zeros(1))
y_pred = tf.matmul(h_relu, w2)

# Compute loss using operations on TensorFlow Tensors
loss = tf.reduce_sum((y - y_pred) ** 2.0)

# Compute gradient of the loss with respect to w1 and w2.
grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

# Update the weights using gradient descent. To actually update the weights
# we need to evaluate new_w1 and new_w2 when executing the graph. Note that
# in TensorFlow the the act of updating the value of the weights is part of
# the computational graph; in PyTorch this happens outside the computational
# graph.
learning_rate = 1e-6
new_w1 = w1.assign(w1 - learning_rate * grad_w1)
new_w2 = w2.assign(w2 - learning_rate * grad_w2)

# Now we have built our computational graph, so we enter a TensorFlow session to
# actually execute the graph.
with tf.Session() as sess:
  # Run the graph once to initialize the Variables w1 and w2.
  sess.run(tf.global_variables_initializer())

  # Create numpy arrays holding the actual data for the inputs x and targets y
  x_value = np.random.randn(N, D_in)
  y_value = np.random.randn(N, D_out)
  for _ in range(500):
    # Execute the graph many times. Each time it executes we want to bind
    # x_value to x and y_value to y, specified with the feed_dict argument.
    # Each time we execute the graph we want to compute the values for loss,
    # new_w1, and new_w2; the values of these Tensors are returned as numpy
    # arrays.
    loss_value, _, _ = sess.run([loss, new_w1, new_w2],
                                feed_dict={x: x_value, y: y_value})
    print(loss_value)
```

### Control Flow and Weight Sharing
As an example of dynamic graphs and weight sharing, we implement a very strange
model: a fully-connected ReLU network that on each forward pass chooses a random
number between 1 and 4 and uses that many hidden layers, reusing the same weights
multiple times to compute the innermost hidden layers.

For this model can use normal Python flow control to implement the loop, and we
can implement weight sharing among the innermost layers by simply reusing the
same Module multiple times when defining the forward pass.

We can easily implement this model as a Module subclass:

```python
# Code in file nn/dynamic_net.py
import random
import torch
from torch.autograd import Variable

class DynamicNet(torch.nn.Module):
  def __init__(self, D_in, H, D_out):
    """
    In the constructor we construct three nn.Linear instances that we will use
    in the forward pass.
    """
    super(DynamicNet, self).__init__()
    self.input_linear = torch.nn.Linear(D_in, H)
    self.middle_linear = torch.nn.Linear(H, H)
    self.output_linear = torch.nn.Linear(H, D_out)

  def forward(self, x):
    """
    For the forward pass of the model, we randomly choose either 0, 1, 2, or 3
    and reuse the middle_linear Module that many times to compute hidden layer
    representations.

    Since each forward pass builds a dynamic computation graph, we can use normal
    Python control-flow operators like loops or conditional statements when
    defining the forward pass of the model.

    Here we also see that it is perfectly safe to reuse the same Module many
    times when defining a computational graph. This is a big improvement from Lua
    Torch, where each Module could be used only once.
    """
    h_relu = self.input_linear(x).clamp(min=0)
    for _ in range(random.randint(0, 3)):
      h_relu = self.middle_linear(h_relu).clamp(min=0)
    y_pred = self.output_linear(h_relu)
    return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs, and wrap them in Variables
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

# Construct our model by instantiating the class defined above
model = DynamicNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. Training this strange model with
# vanilla stochastic gradient descent is tough, so we use momentum
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
for t in range(500):
  # Forward pass: Compute predicted y by passing x to the model
  y_pred = model(x)

  # Compute and print loss
  loss = criterion(y_pred, y)
  print(t, loss.data[0])

  # Zero gradients, perform a backward pass, and update the weights.
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
```
