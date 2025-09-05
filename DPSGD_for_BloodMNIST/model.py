import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.GroupNorm(4, 16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.GroupNorm(4, 16), #nn.BatchNorm2d(16)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.GroupNorm(8, 64),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.GroupNorm(8, 64),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64), #  nn.BatchNorm2d(64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    


def get_model(n_channels, n_classes, device=None):
    """
    Instantiate and return the model, moving it to GPU if available.
    
    Args:
        n_channels: Number of input channels
        n_classes: Number of output classes
        device: Optional device to use (if None, will use CUDA if available)
        
    Returns:
        Model placed on the appropriate device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Net(n_channels, n_classes)
    model = model.to(device)
    
    print(f"Using device: {device}")
    
    return model

def get_optimizer(model, lr=0.001):
    """Return Adam optimizer for the model."""
    return optim.Adam(model.parameters(), lr=lr)

def get_loss_function(task):
    """Return appropriate loss function based on task type."""
    if task == 'multi-class':
        return nn.CrossEntropyLoss()
    elif task == 'multi-label':
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown task: {task}")

def load_model(model_path, device, n_channels = 3,  n_classes =  8):
    """
    Load a model from a given path. 
    
    Args:
        model_path (str): path to the saved model weights
        n_channels (int): number of input channels (3)
        n_classes (int): number of output classes (8)
    
    Returns:
        torch.nn.Module: model
    """
    model = get_model(n_channels, n_classes)
    state_dict = torch.load(model_path, map_location = device)
    model.load_state_dict(state_dict)
    return model 