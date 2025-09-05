import torch
from torch.utils.data import DataLoader
import numpy as np
from adagradnorm import AdagradNorm


def select_optimizer(model, train_dataset, optimizer_name="sgd", batch_type="mini-batch", 
                              batch_size=64, set_lr = False, learning_rate=0.01):
 
    """
    Creates and return an optimizer and data loader based on specified parameters.
    
    Args:
        model: the model whose parameters will be optimized
        train_dataset: the training dataset
        optimizer_name: name of the optimizer to use ('sgd', 'adam', 'lbfgs', 'rmsprop', 'adagrad')
        batch_type: 'mini-batch' or 'full' batch training
        batch_size: size of mini-batches (ignored if batch_type is 'full')
        set_lr: whether to use the provided learning rate
        learning_rate (float): learning rate to use if set_lr is True

    Returns:
        tuple: (optimizer, train_loader) 
    """

    if optimizer_name.lower() == "sgd" or optimizer_name.lower() == "gd":
        if set_lr:
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.SGD(model.parameters()) #default: LR=0.001
        
    elif optimizer_name.lower() == "lbfgs":
        batch_type = "full"
        if set_lr:
            optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.LBFGS(model.parameters()) #default: LR=0.01
        
    elif optimizer_name.lower() == "adam":
        if set_lr:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.Adam(model.parameters()) # default: LR=0.001
        
    elif optimizer_name.lower() == "rmsprop":
        if set_lr:
            optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)# default: LR=0.01 -> changed to the recommended 0.001
            
    elif optimizer_name.lower() == "adagrad":
        if set_lr:
            optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.Adagrad(model.parameters())# default: LR=0.01

    elif optimizer_name.lower() == "adagrad_norm":
        if set_lr:
            optimizer = AdagradNorm(model.parameters(),lr=learning_rate)
        else: 
            optimizer = AdagradNorm(model.parameters())
            
    else:
        raise ValueError(f"We don't consider {optimizer_name}")

    if batch_type == "full": 
        train_loader = DataLoader(
            train_dataset, 
            batch_size=len(train_dataset),
            shuffle=False
        )
    else:  
        # default: mini-batch
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True
        )
    
    return optimizer, train_loader



def train_epoch_lbfgs(model, opt, loss_fun, closure, X_test_tensor, y_test_tensor, device, is_binary=True):
    """Trains 1 epoch using L-BFGS optimizer"""
    loss = opt.step(closure)
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor.to(device))
        test_loss = loss_fun(test_outputs, y_test_tensor.to(device)).item()
        test_accuracy = acc(test_outputs, y_test_tensor.to(device), is_binary)
        
    return loss.item(), test_loss, test_accuracy

def train_epoch_sgd(model, opt, loss_fun, train_loader, test_loader, device, scheduler = None, is_binary=True):
    """Trains 1 epoch using SGD-based optimizers"""
    model.train()
    train_loss = sum(
        train_batch(model, opt, loss_fun, inputs.to(device), targets.to(device))
        for inputs, targets in train_loader
    )

    if scheduler is not None:
        scheduler.step()
    
    model.eval()
    with torch.no_grad():
        test_metrics = [
            evaluate_batch(model, loss_fun, inputs.to(device), targets.to(device), is_binary)
            for inputs, targets in test_loader
        ]
    
    avg_train_loss = train_loss / len(train_loader)
    avg_test_loss = sum(loss for loss, _ in test_metrics) / len(test_metrics)
    avg_test_acc = sum(acc for _, acc in test_metrics) / len(test_metrics)
    
    return avg_train_loss, avg_test_loss, avg_test_acc

def train_batch(model, opt, loss_fun, inputs, targets):
    """Trains a single batch"""
    opt.zero_grad()
    outputs = model(inputs)
    loss = loss_fun(outputs, targets)
    loss.backward()
    opt.step()
    return loss.item()

def evaluate_batch(model, loss_fun, inputs, targets, is_binary=True):
    """Evaluates a single batch"""
    outputs = model(inputs)
    loss = loss_fun(outputs, targets)
    accuracy = acc(outputs, targets, is_binary)
    return loss.item(), accuracy
    
# Metric
def acc(outputs, targets, is_binary=True):

    if is_binary:
        accuracy = binary_acc(outputs, targets)
    else:
        predictions = torch.argmax(outputs, dim=1)
        accuracy = float((predictions == targets).float().mean())
    
    return accuracy

def binary_acc(outputs, targets):
    """
    Calculates binary classification accuracy.
    
    Args:
        outputs: model outputs (probabilities)
        targets: ground truth binary labels

    Returns:
        float: binary classification accuracy
    """
    predictions = (outputs >= 0.5).float()  
    return float((predictions == targets).float().mean())

def baseline(n_classes, num_samples, train_loader):
    """
    Calculates a baseline prediction based on class distribution.

    Args:
        n_classes: number of classes
        num_samples: number of samples to generate predictions for
        train_loader: training data loader to compute class distribution

    Returns:
        numpy.ndarray: baseline predictions 
    """
    class_counts = torch.zeros(n_classes, dtype=torch.int64)
    total_count = 0

    # to count each class in the training dataset
    for _, labels in train_loader:
        for i in range(n_classes):
            class_counts[i] += (labels == i).sum()
        total_count += len(labels)
    
    # calculate class probabilities
    class_probs = class_counts.float() / total_count

    return np.tile(class_probs.numpy(), (num_samples, 1))