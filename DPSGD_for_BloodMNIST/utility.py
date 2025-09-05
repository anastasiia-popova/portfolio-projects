import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import medmnist
from medmnist import INFO, Evaluator

def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_data_info(data_flag):
    """Fetch dataset info from MedMNIST."""
    info = INFO[data_flag]
    task = info['task']
    return info

def get_dataclass(info):
    """Return the MedMNIST dataset class."""
    return getattr(medmnist, info['python_class'])

def get_data_transforms():
    """Return the torchvision transforms for preprocessing."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

def get_datasets(data_flag, download=True):
    """Load train, val, and test datasets with transforms."""
    info = get_data_info(data_flag)
    DataClass = get_dataclass(info)
    data_transform = get_data_transforms()
    train_dataset = DataClass(split='train', transform=data_transform, download=download)
    val_dataset = DataClass(split='val', transform=data_transform, download=download)
    test_dataset = DataClass(split='test', transform=data_transform, download=download)
    return train_dataset, val_dataset, test_dataset

def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    """Create PyTorch DataLoaders for the datasets."""
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def scores(split, model, info, data_flag, train_loader_at_eval, val_loader_at_eval, test_loader, return_results=True):
    """
    Evaluate model performance on the specified data split.
    
    Args:
        split: Which data split to evaluate on ('train', 'val', 'test')
        model: PyTorch model to evaluate
        info: Dataset information dictionary
        data_flag: MedMNIST dataset flag
        train_loader_at_eval: DataLoader for training data evaluation
        val_loader_at_eval: DataLoader for validation data
        test_loader: DataLoader for test data
        return_results: If True, return metrics; otherwise, print them
        
    Returns:
        If return_results is True, returns metrics (auc, acc)
    """

    
    model.eval()
    device = next(model.parameters()).device  # Get model's device
    
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)

    if split == 'train':
        data_loader = train_loader_at_eval
    elif split == 'val':
        data_loader = val_loader_at_eval
    else:
        data_loader = test_loader

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)

            if info['task'] == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                outputs = outputs.softmax(dim=-1)
            else:
                targets = targets.squeeze().long()
                outputs = outputs.softmax(dim=-1)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        
        evaluator = Evaluator(data_flag, split)
        metrics = evaluator.evaluate(y_score)

        if return_results:
            return metrics
        else:
            print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))




def get_single_prediction(model, test_dataset, index=None):
    """
    Get prediction for a single example from the test dataset
    
    Args:
        model: the trained model
        test_dataset: the test dataset
        index: index of the example to use, random if None
    
    Returns:
        true_label: label
        image: the input image
        pred_label: predicted label
        pred_probs: prediction probabilities for all classes
    """
    model.eval()
    device = next(model.parameters()).device  # Get model's device
    
    # get random image index 
    if index is None:
        index = np.random.randint(0, len(test_dataset))
        
    image, label = test_dataset[index]
    image_tensor = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    return label.item(), image, predicted_class, probabilities[0].cpu().numpy()

def get_test_acc(model, test_loader, info, data_flag):
    """
    Compute AUC and ACC on the test set for a given model.
    
    Args:
        model (torch.nn.Module): the trained model
        test_loader: DataLoader for test data
        info: Dataset information dictionary
        data_flag: MedMNIST dataset flag
    
    Returns:
        tuple: (AUC, ACC)
    """
    metric_test = scores('test', model, info, data_flag, test_loader, test_loader, test_loader) 
    return metric_test.AUC, metric_test.ACC 

def denormalize(img, mean=[0.5]*3, std=[0.5]*3):
    """
    Revert normalization for an image tensor.
    
    Args:
        img (torch.Tensor): normalized image tensor with dims (C, H, W)
        mean (list): mean used for normalization
        std (list): std used for normalization
    
    Returns:
        torch.Tensor: original image tensor
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return img*std + mean