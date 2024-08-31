# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 12:18:16 2024

@author: Kirill
"""

    
##########################################################
# Replication Code for Neural Network Binary Classifiers #
# used in the 2024 paper "Applying Machine Learning      #
# to Election Forensics Research: A Case of Russia"      #
##########################################################


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import f1_score


def sigmoid_focal_loss(inputs, targets, weights=None, alpha=0.25, gamma=2.0, reduction='mean'):
    
    bce_loss = nn.BCELoss(reduction='none')(inputs, targets)
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
     
    if weights is not None:
        focal_loss = focal_loss * weights  # Apply weights

    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss
    
def generate_region_weights(region_series, fraud_series, weight_by_fraud=True):
    if weight_by_fraud:
        # Calculate the number of fraudulent cases per region
        region_fraud_counts = fraud_series.groupby(region_series).sum()
        
        # Invert the count to get higher weights for regions with fewer frauds
        weights = 1.0 / (region_fraud_counts + 1e-2)  # Add small epsilon to avoid division by zero
    else:
        # Calculate the number of samples per region
        region_counts = region_series.value_counts()
        total_samples = len(region_series)
        
        # Calculate weights proportional to the inverse of region frequency
        weights = total_samples / (len(region_counts) * region_counts)
    
    # Normalize the weights if needed (optional)
    weights = weights / weights.sum()

    # Map the weights back to the original series
    weights_series = region_series.map(weights.to_dict())

    return torch.tensor(weights_series.values, dtype=torch.float32)


class SmallNN(nn.Module):
    def __init__(self, input_size):
        super(SmallNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 20)
        self.fc2 = nn.Linear(20, 30)
        self.fc3 = nn.Linear(30, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

class AverageNN(nn.Module):
    def __init__(self, input_size):
        super(AverageNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 20)
        self.fc2 = nn.Linear(20, 30)
        self.fc3 = nn.Linear(30, 50)
        self.fc4 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

class LargeNN(nn.Module):
    def __init__(self, input_size):
        super(LargeNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 20)
        self.fc2 = nn.Linear(20, 30)
        self.fc3 = nn.Linear(30, 50)
        self.fc4 = nn.Linear(50, 100)
        self.fc5 = nn.Linear(100, 100)
        self.fc6 = nn.Linear(100, 50)
        self.fc7 = nn.Linear(50, 50)
        self.fc8 = nn.Linear(50, 50)
        self.fc9 = nn.Linear(50, 10)
        self.fc10 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        x = torch.relu(x)
        x = self.fc5(x)
        x = torch.relu(x)
        x = self.fc6(x)
        x = torch.relu(x)
        x = self.fc7(x)
        x = torch.relu(x)
        x = self.fc8(x)
        x = torch.relu(x)
        x = self.fc9(x)
        x = torch.relu(x)
        x = self.fc10(x)
        x = self.sigmoid(x)
        return x


def plot_losses(train_losses, test_losses, f1_scores, save_as_png=False, filename="loss_plot.png"):
    # Plot 1: Train and Test Losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Epochs vs. Train/Test Losses')
    plt.legend()
    if save_as_png:
        plt.savefig(filename.replace(".png", "_losses.png"))
    plt.show()

    # Plot 2: F1 Scores
    plt.figure(figsize=(10, 5))
    plt.plot(f1_scores, label='Test F1 Score', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('Epochs vs. F1 Score')
    plt.legend()
    if save_as_png:
        plt.savefig(filename.replace(".png", "_f1_score.png"))
    plt.show()
    


def subset(dat):
    # Create a new column that combines 'regname' and 'commission'
    dat['reg_territ'] = dat['regname'] + ':' + dat['commission']
    # Select only region-territories for which fraud is known
    reg_territ_fraud = dat.loc[dat['fraud'] == 1, 'reg_territ'].unique()
    # Filter the DataFrame based on 'reg_territ_fraud'
    dat_sel = dat[dat['reg_territ'].isin(reg_territ_fraud)]
    return dat_sel


  
def train_neural_network(X, y, region, model_class, oversampling_method='none', 
                         num_epochs=2, lr=0.001, alpha=0.25, gamma=2.0, 
                         loss_function='focal', use_weights=True,
                         estimate_fraud=True, plot_loss=True, filename=None):

    # Ensure target is binary and in range [0, 1]
    y = (y == 1).astype(np.float32)

    # Combine features and target for resampling
    X = pd.DataFrame(X).reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    # Generate region weights if needed
    if use_weights:
        weights = generate_region_weights(region, y, weight_by_fraud=True)
        weights = pd.Series(weights.numpy())
        weights_df = pd.DataFrame(weights, columns=['weights'])
        df_combined = pd.concat([pd.DataFrame(X).reset_index(drop=True), 
                                 pd.Series(y, name='fraud').reset_index(drop=True), 
                                 weights_df], axis=1)
    else:
        weights = None
        df_combined = pd.concat([pd.DataFrame(X).reset_index(drop=True), 
                                 pd.Series(y, name='fraud').reset_index(drop=True)], axis=1)        

    # Oversampling
    if oversampling_method == 'oversample':
        df_majority = df_combined[df_combined[y.name] == 0]
        df_minority = df_combined[df_combined[y.name] == 1]
        df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=123)
        df_upsampled = pd.concat([df_majority, df_minority_upsampled])
        X_upsampled = df_upsampled.drop(y.name, axis=1)
        y_upsampled = df_upsampled[y.name]
        
    elif oversampling_method == 'smote':
        smote = SMOTE(random_state=123)
        X_upsampled, y_upsampled = smote.fit_resample(X, y)
    else:
        X_upsampled = pd.concat([pd.DataFrame(X).reset_index(drop=True),  weights_df], axis=1)
        y_upsampled = y

    # Split data into train and test sets
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(
        X_upsampled, y_upsampled, test_size=test_size, stratify=y_upsampled, random_state=0
    )
    
    if use_weights:
        weights_train = X_train["weights"]
        weights_test = X_test["weights"]
        X_train = X_train.drop("weights", axis=1)
        X_test = X_test.drop("weights", axis=1)

    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    #model = SimpleNN(X_train.shape[1])
    model = model_class(X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []
    f1_scores = []
        
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()

            if loss_function == 'focal':
                if use_weights:
                    start_idx = batch_idx * train_loader.batch_size
                    end_idx = start_idx + inputs.size(0)
                    indices = range(start_idx, end_idx)
                    batch_weights = weights_train.iloc[list(indices)].to_numpy()
                    batch_weights = torch.tensor(batch_weights, dtype=torch.float32, device=inputs.device)
                else:
                    batch_weights = None
                loss = sigmoid_focal_loss(outputs, targets, weights=batch_weights, alpha=alpha, gamma=gamma, reduction='mean')
            elif loss_function == 'cross_entropy':
                loss = F.binary_cross_entropy(outputs, targets, reduction='mean')
            else:
                raise ValueError(f"Unsupported loss function: {loss_function}")

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
            
        model.eval()
        test_loss = 0.0
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                outputs = model(inputs).squeeze()
                predictions = (outputs > 0.5).float()

                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

                if loss_function == 'focal':
                    if use_weights:
                        start_idx = batch_idx * test_loader.batch_size
                        end_idx = start_idx + inputs.size(0)
                        indices = range(start_idx, end_idx)
                        batch_weights = weights_test.iloc[list(indices)].to_numpy()
                        batch_weights = torch.tensor(batch_weights, dtype=torch.float32, device=inputs.device)
                    else:
                        batch_weights = None
                    loss = sigmoid_focal_loss(outputs, targets, weights=batch_weights, alpha=alpha, gamma=gamma, reduction='mean')
                elif loss_function == 'cross_entropy':
                    loss = F.binary_cross_entropy(outputs, targets, reduction='mean')
                else:
                    raise ValueError(f"Unsupported loss function: {loss_function}")

                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        f1_weighted = f1_score(all_targets, all_predictions, average='weighted')
        f1_scores.append(f1_weighted)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}, Test Loss: {avg_test_loss}, F1 Weighted Score: {f1_weighted}')

    if estimate_fraud:
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test).squeeze()
            test_predictions = (test_outputs > 0.5).float()  # Binary classification threshold
            test_fraud_proportion = test_predictions.mean().item()

            print(f'Proportion of fraud in the test set: {test_fraud_proportion}')
        
            test_outputs = model(torch.tensor(X_test, dtype=torch.float32)).squeeze()
            test_predictions = (test_outputs > 0.5).float()  # Binary classification threshold
            test_fraud_precincts = test_predictions.numpy()
            fraud_predictions_series = pd.Series(test_fraud_precincts, name='fraud_predictions')
            region = region.reset_index(drop=True)
            fraud_predictions_series = fraud_predictions_series.reset_index(drop=True)
            test_precinct_fraud_df = pd.concat([region, fraud_predictions_series], axis=1)
            test_region_fraud_mean = test_precinct_fraud_df.groupby(region.name)['fraud_predictions'].mean().reset_index()

        model.eval()
        with torch.no_grad():
            all_outputs = model(torch.tensor(X.values, dtype=torch.float32)).squeeze()
            all_predictions = (all_outputs > 0.5).float()  # Binary classification threshold
            all_fraud_proportion = all_predictions.mean().item()
            
            print(f'Proportion of fraud in the entire set: {all_fraud_proportion}')
            
            all_outputs = model(torch.tensor(X.values, dtype=torch.float32)).squeeze()
            all_predictions = (all_outputs > 0.5).float()  # Binary classification threshold
            all_fraud_precincts = all_predictions.numpy()
            fraud_predictions_series = pd.Series(all_fraud_precincts, name='fraud_predictions')
            region = region.reset_index(drop=True)
            all_fraud_predictions_series = fraud_predictions_series.reset_index(drop=True)
            all_precinct_fraud_df = pd.concat([region, all_fraud_predictions_series], axis=1)
            all_region_fraud_mean = all_precinct_fraud_df.groupby(region.name)['fraud_predictions'].mean().reset_index()
             
            predictions_series = pd.Series(all_predictions.numpy(), name='predictions')
            
        estimated_fraud = {
            'test_fraud_proportion': test_fraud_proportion,
            'all_fraud_proportion': all_fraud_proportion,
            'test_region_fraud_mean': test_region_fraud_mean,
            'all_region_fraud_mean': all_region_fraud_mean,
            'all_predictions': predictions_series}
        
    if plot_loss:
        filename = filename if filename is not None else "loss_plot.png"
        plot_losses(train_losses, test_losses, f1_scores, save_as_png=True, filename=filename)
        
    return train_losses, test_losses, f1_scores, model, estimated_fraud
        


#Example 
loss_function='focal'
num_epochs=50
lr = 0.005
oversampling_method='oversample'
dat = pd.read_csv('data/election2012.csv')
dat = dat.fillna(0)
X = dat.iloc[:, list(range(13, 36)) + list(range(37, 50)) + list(range(54, 56)) + list(range(57, 63)) + list(range(65, 69))]
X = normalize(X.to_numpy(), axis=0); y = dat.iloc[:, -1]
results2012 = train_neural_network(X, y, region=dat['regname'], model_class=SmallNN, oversampling_method=oversampling_method, 
                                   loss_function=loss_function, num_epochs=num_epochs, lr=lr,
                                   estimate_fraud=True, plot_loss=True, use_weights=True, 
                                   filename="SmallNN_plot2012_focal.png")

precinct_result = results2012[4]['all_predictions']
dat['estfraud'] = precinct_result
dat['regname2'] = dat['regname'].map(region_mapping)
dat2 = dat[['regname2', 'uik', 'estfraud', 'fraud']]
dat2.to_csv('dat2012_results.csv', index=False)


