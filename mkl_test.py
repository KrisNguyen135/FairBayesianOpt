# -*- coding: utf-8 -*-
import torch

N, D_in, D_out, H, W = 64, 1000, 100, 128, 128

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in, H, W)
y = torch.randn(N, D_out, H, W)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output.
model = torch.nn.Sequential(
    torch.nn.Conv2d(D_in, D_out, 3, padding=1),
    torch.nn.BatchNorm2d(D_out),
    torch.nn.ReLU(),
)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
for t in range(5):
    # Forward pass: compute predicted y by passing x to the model. Module
    # objects override the __call__ operator so you can call them like
    # functions. When doing so you pass a Tensor of input data to the Module
    # and it producesa Tensor of output data.
    y_pred = model(x)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the
    # learnable parameters of the model. Internally, the parameters of each
    # Module are stored in Tensors with requires_grad=True, so this call will
    # compute gradients for all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
