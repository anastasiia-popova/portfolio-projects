#!/usr/bin/env python
# coding: utf-8
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim # (SGD), Adam, RMSprop, optimizer.step() 
import torch.utils.data as data 

import torchvision.transforms as transforms # suite of tools for image preprocessing and data augmentation, commonly used in computer vision tasks.
# transforms.Compose to build complex preprocessing pipelines 
# Supports both PIL images and PyTorch tensors.
import json
import csv
from pathlib import Path
from opacus import PrivacyEngine

import os
import time 


from utility import set_seed, get_datasets, get_dataloaders, get_data_info, scores
from model import get_model, get_optimizer, get_loss_function 

experiment_id = 'effect_of_batch_size'  # Unique identifier for this experiment

use_google_colab = True 

if use_google_colab:
    # Install MedMNIST and privacy tools
    # !pip install medmnist fire
    # !pip install opacus==1.4.0 # isn't used in this script, but can be useful for private training
    # from google.colab import drive
    # drive.mount('/content/drive')
    # ! python dpsgd_training.py

    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'Not available'}")

    DRIVE_BASE_PATH = '/content/drive/MyDrive/BloodMNIST_Experiments' 
    os.makedirs(DRIVE_BASE_PATH, exist_ok=True)


    MODEL_SAVE_PATH = f'{DRIVE_BASE_PATH}/weights/{experiment_id}'
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    OUTPUT_CSV_PATH = f'{DRIVE_BASE_PATH}/output_{experiment_id}.csv'
    METADATA_JSON_PATH = f'{DRIVE_BASE_PATH}/metadata_{experiment_id}.json'
else:

    MODEL_SAVE_PATH = f'weights/{experiment_id}'
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    OUTPUT_CSV_PATH = f'output_{experiment_id}.csv'
    METADATA_JSON_PATH = f'metadata_{experiment_id}.json'

data_flag = 'bloodmnist'
info = get_data_info(data_flag)

# ## Main Training Parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)  # For reproducibility
# Training parameters
NUM_EPOCHS = 100
LR = 0.00127
batch_sizes = [1024, 512, 256, 128, 64, 32, 16]

n_channels = info['n_channels']
n_classes = len(info['label'])
K_max = 3  # Number of initializations for averaging results

# Early stopping parameters
PATIENCE = 10  # number of epochs we wait for improvement
MIN_DELTA = 0.01  # minimum improvement
MIN_EPOCHS = 40  # minimum number of epochs before early stopping can be triggered, must be less than NUM_EPOCHS
if MIN_EPOCHS >= NUM_EPOCHS:
    raise ValueError("MIN_EPOCHS must be less than NUM_EPOCHS for early stopping to work.")


# Privacy Parameters 
C = 20.6914 # find_clipping: list(np.logspace(-2, 1.5, num=20))
DELTA = 7*10**(-9) # must have 1/N^2 
EPSILON = 2 # effect_of_epsilon: list(np.logspace(-2, 2, num=20)) 

# Initialize base metadata structure
meta_data = {
    'experiment_id': experiment_id,
    'data_flag': data_flag,
    'num_epochs': NUM_EPOCHS,
    'num_initializations': K_max,
    'patience': PATIENCE,
    'min_delta': MIN_DELTA,
    'min_epochs': MIN_EPOCHS,
    'batch_sizes': batch_sizes,
    'clipping_param': C,
    'delta': DELTA,
    'epsilon': EPSILON,
    'used_device': str(device),
    'learning_rate': float(LR),
    'batch_results': {}  
}


fieldNames = ['batch_size', 'init_id', 'best_epoch', 'early_stop_time', 'train_acc', 'train_auc', 'test_acc', 'test_auc', 'avg_epoch_time', 'std_epoch_time']

# Create CSV with updated headers
if not os.path.exists(OUTPUT_CSV_PATH):
    with open(OUTPUT_CSV_PATH, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldNames)
        writer.writeheader()

# Loop through parameters
for batch_size in batch_sizes:

    # ## Preprocess and dataloader

    train_dataset, val_dataset, test_dataset = get_datasets(data_flag)
    train_loader, val_loader, test_loader = get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=batch_size)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*batch_size, shuffle=False)
    val_loader_at_eval = data.DataLoader(dataset=val_dataset, batch_size=2*batch_size, shuffle=False)

    print(f"Training with batch_size = {batch_size}")
    train_accs = []
    train_aucs = []
    test_accs = []
    test_aucs = []
    early_stop_epochs = []
    early_stop_times = []
    best_models = []
    epoch_times = []
    
    for K in range(K_max):
        print(f"Initialization {K+1}/{K_max}")
        # need to init model weights 
        model = get_model(n_channels, n_classes)
        optimizer = get_optimizer(model, lr=LR)
        criterion = get_loss_function(info['task'])

        # Initialize PrivacyEngine 
        privacy_engine = PrivacyEngine()
        # old version of Opacus for final model: privacy_engine = PrivacyEngine(secure_mode=True)

        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=EPSILON,
            target_delta = DELTA,
            epochs = NUM_EPOCHS,
            max_grad_norm=C,     # Gradient clipping norm 
            loss_reduction = 'mean', #Indicates if the loss reduction (for aggregating the gradients) is a sum or a mean operation.
        )

        # Early stopping variables
        best_val_metric = 0  # For ACC (higher is better)
        best_epoch = 0
        no_improve_count = 0
        best_model_path = None

        # Time tracking
        total_time = 0
        epoch_times = []
        
        # For tracking metrics
        loss_curve_train = []
        loss_curve_val = []
        

        for epoch in range(NUM_EPOCHS):
            train_loss = 0.0
            train_batches = 0
        
            # Training phase
            start_epoch_time = time.perf_counter() 

            model.train()        
            for inputs, targets in tqdm(train_loader):

                inputs = inputs.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)

                # can be used for other 2D datasets 
                if info['task'] == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                    loss = criterion(outputs, targets)
                else:
                    targets = targets.squeeze().long()
                    loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
        
                train_loss += loss.item()
                train_batches += 1
        
            avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
            loss_curve_train.append(avg_train_loss)
    
            # Validation 
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                
                for val_inputs, val_targets in val_loader:
                    val_inputs = val_inputs.to(device)
                    val_targets = val_targets.to(device)
                    
                    val_outputs = model(val_inputs)
                    if info['task'] == 'multi-label, binary-class':
                        val_targets = val_targets.to(torch.float32)
                        v_loss = criterion(val_outputs, val_targets)
                    else:
                        val_targets = val_targets.squeeze().long()
                        v_loss = criterion(val_outputs, val_targets)
                        
                    val_loss += v_loss.item()
                    val_batches += 1
                    
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
            loss_curve_val.append(avg_val_loss)

            # ---- Calculate validation ACC for early stopping ----
            metric_val = scores('val', model, info, data_flag, train_loader_at_eval, val_loader_at_eval, test_loader)

            current_metric = metric_val.ACC
            end_epoch_time = time.perf_counter()
            epoch_time = end_epoch_time - start_epoch_time
            epoch_times.append(epoch_time)
            total_time += epoch_time
                    
            print(f"Epoch {epoch+1} train loss: {avg_train_loss:.4f}, val loss: {avg_val_loss:.4f}")
            print(f"Epoch {epoch+1} training time: {epoch_time:.2f} seconds")
            
            # Early stopping check
            if epoch > MIN_EPOCHS and current_metric > best_val_metric + MIN_DELTA:
                print(f"Validation metric improved from {best_val_metric:.4f} to {current_metric:.4f}")
                best_val_metric = current_metric
                best_epoch = epoch
                no_improve_count = 0
                
                # Save best model with updated path
                model_filename = f'model_init_{K}_best_epoch_{epoch}_auc_{metric_val.AUC:.4f}_acc_{metric_val.ACC:.4f}.pth'
                model_path = os.path.join(MODEL_SAVE_PATH, model_filename)

                # Check if model has the _module attribute (from privacy engine)
                if hasattr(model, '_module'):
                    torch.save(model._module.state_dict(), model_path)
                else:
                    torch.save(model.state_dict(), model_path)

                best_model_path = model_path
                print(f"Saved best model to {model_path}")
            else:
                if epoch > MIN_EPOCHS:
                    no_improve_count += 1
                    print(f"No improvement for {no_improve_count} epochs")
                
            # Check if we should stop
            if epoch > MIN_EPOCHS and no_improve_count >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
        

        # Check if a best model was actually saved (for debugging)
        if best_model_path is None:
            print(f"Warning: No model was saved for initialization {K}. Using the final model instead.")
            # Save the final model state
            model_filename = f'model_init_{K}_final_epoch_{epoch}.pth'
            model_path = os.path.join(MODEL_SAVE_PATH, model_filename)
            # This gets the original model without the _module prefix
            if hasattr(model, '_module'):
                torch.save(model._module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
            best_model_path = model_path
            best_models.append(best_model_path)

        # After training is done for this initialization, evaluate the model
        model = get_model(n_channels, n_classes)
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        
        metric_train = scores('train', model, info, data_flag, train_loader_at_eval, val_loader_at_eval, test_loader)
        metric_test = scores('test', model, info, data_flag, train_loader_at_eval, val_loader_at_eval, test_loader)
        

        avg_epoch_time_value = np.mean(epoch_times) if epoch_times else 0
        std_epoch_time_value = np.std(epoch_times) if epoch_times else 0

        train_accs.append(metric_train.ACC)
        train_aucs.append(metric_train.AUC)
        test_accs.append(metric_test.ACC)
        test_aucs.append(metric_test.AUC)
        early_stop_epochs.append(best_epoch + 1)  # +1 for 1-based indexing
        early_stop_times.append(total_time)
        best_models.append(best_model_path)
        epoch_times.extend(epoch_times)  # Add all epoch times for calculating overall stats
        
        
        # Update CSV with results for the current initialization
        with open(OUTPUT_CSV_PATH, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldNames)
            if csvfile.tell() == 0:  # File is empty
                writer.writeheader()
            writer.writerow({
                'batch_size': batch_size,
                'init_id': K,
                'best_epoch': best_epoch + 1,
                'early_stop_time': total_time,
                'train_acc': metric_train.ACC,
                'train_auc': metric_train.AUC,
                'test_acc': metric_test.ACC,
                'test_auc': metric_test.AUC,
                'avg_epoch_time': avg_epoch_time_value,
                'std_epoch_time': std_epoch_time_value
            })
            csvfile.flush()  # Force write to disk
            os.fsync(csvfile.fileno())  # Force OS to write to physical storage
    
    
    # Store aggregate results
    meta_data['batch_results'][str(batch_size)] = {
        'train_acc_mean': float(np.mean(train_accs)),
        'train_acc_std': float(np.std(train_accs)),
        'train_auc_mean': float(np.mean(train_aucs)),
        'train_auc_std': float(np.std(train_aucs)),
        'test_acc_mean': float(np.mean(test_accs)),
        'test_acc_std': float(np.std(test_accs)),
        'test_auc_mean': float(np.mean(test_aucs)),
        'test_auc_std': float(np.std(test_aucs)),
        'early_stop_epochs_mean': float(np.mean(early_stop_epochs)),
        'early_stop_epochs_std': float(np.std(early_stop_epochs)),
        'training_time_mean': float(np.mean(early_stop_times)),
        'training_time_std': float(np.std(early_stop_times)),
        'epoch_time_mean': float(np.mean(epoch_times)),
        'epoch_time_std': float(np.std(epoch_times)),
        'best_models': best_models
    }


# Save final metadata to JSON (it is better to change and don't track statistics in metadata) 
with open(METADATA_JSON_PATH, 'w') as f:
    json.dump(meta_data, f, indent=4)

print(f"Metadata saved to {METADATA_JSON_PATH}")
print(f"Output data saved to {OUTPUT_CSV_PATH}")






