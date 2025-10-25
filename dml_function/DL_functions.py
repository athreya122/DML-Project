# functions #

# HEADER FILES
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler



# DATA HANDLING FUNCTIONS
#def pad_sequence(sequences):
#    pad_sequence(sequences,batch_first=True)  # batch_first is having (B x T x *) rather than (T x B x *), where T is the longest sequence


# DATA PADDING
class DS():
    def __init__(self,X,y):
        self.X = torch.tensor(X,dtype = torch.float32)
        self.y = torch.tensor(y,dtype = torch.float32)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        return self.X[idx], self.y[idx]


    

# NODE DEFINITION
class GRU_Node(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        Inputs:
            input_size     - Dimensionality of the input of this RNN.
            hidden_size    - Dimensionality of the hidden state.
            output_size    - Dimensionality of the output of this RNN.
        """

        super(GRU_Node, self).__init__()
        self.hidden_size = hidden_size
        
        self.Wz = nn.Linear(input_size,hidden_size)
        self.Uz = nn.Linear(hidden_size,hidden_size,bias = False)

        self.Wr = nn.Linear(input_size,hidden_size)
        self.Ur = nn.Linear(hidden_size,hidden_size,bias = False)

        self.Wh = nn.Linear(input_size,hidden_size)
        self.Uh = nn.Linear(hidden_size,hidden_size,bias = False)

    def forward(self,x,h_prev):
        # Npdate
        zt = torch.sigmoid(self.Wz(x) + self.Uz(h_prev))
        # reset
        rt = torch.sigmoid(self.Wr(x) + self.Ur(h_prev))
        # cand. hidden
        h_tilde = torch.tanh(self.Wh(x) + self.Uh(rt*h_prev))
        #ht
        ht = (1-zt)*h_prev + zt*h_tilde
        
        return ht


# LAYER INCLUDING FINAL FC
class GRULayer(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(GRULayer,self).__init__()
        self.hidden_size = hidden_size
        self.gru_node = GRU_Node(input_size, hidden_size)
        # final fully connected layer
        self.fc = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        #x_shape = (batch_size,seqeunce_length(no. of data look back on),input_size(or feature size))
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size,self.hidden_size)

        h_vals = []
        for t in range(seq_len):
            xt = x[:, t, :]
            h = self.gru_node(xt, h)
            h_vals.append(h.unsqueeze(1))

        h_vals = torch.cat(h_vals, dim=1)
        output = self.fc(h)
        return h_vals, h, output
    
# class GRULayer(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.2):
#         super(GRULayer, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.dropout = dropout

#         self.gru_nodes = nn.ModuleList([
#             GRU_Node(input_size if i == 0 else hidden_size, hidden_size)
#             for i in range(num_layers)
#         ])

#         self.dropouts = nn.ModuleList([
#             nn.Dropout(dropout) for _ in range(num_layers - 1)
#         ])

#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         batch_size, seq_len, _ = x.size()
#         h = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]

#         h_vals = []
#         for t in range(seq_len):
#             xt = x[:, t, :]
#             for i, gru_node in enumerate(self.gru_nodes):
#                 xt = gru_node(xt, h[i])
#                 h[i] = xt
#                 if i < self.num_layers - 1:
#                     xt = self.dropouts[i](xt)
#             h_vals.append(xt.unsqueeze(1))

#         h_vals = torch.cat(h_vals, dim=1)
#         output = self.fc(h[-1])
#         return h_vals, h[-1], output

    
# TRAINING METHOD DEFINITION
def train_n_validate(model, train_loader, val_loader, num_epochs, optimizer, loss_criterion, device):
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            _, _, y_pred = model(X_batch)

            loss = loss_criterion(y_pred.view(-1), y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}", end="")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                _, _, y_pred = model(X_batch)
                val_loss = loss_criterion(y_pred.view(-1), y_batch)
                val_loss_sum += val_loss.item()
                
            avg_val_loss = val_loss_sum / len(val_loader)
            val_losses.append(avg_val_loss)
            print(f" | Val Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses



# Additional functions for the multiple battery

#scaler functions
def scale_battery(train_df, test_df, feature_cols):
    scalers = {c: MinMaxScaler().fit(train_df[[c]]) for c in feature_cols}
    for c in feature_cols:
        train_df[c] = scalers[c].transform(train_df[[c]])
        test_df[c] = scalers[c].transform(test_df[[c]])

    target_scaler = MinMaxScaler().fit(train_df[["capacity"]])
    train_df["capacity"] = target_scaler.transform(train_df[["capacity"]])
    test_df["capacity"] = target_scaler.transform(test_df[["capacity"]])

    return train_df, test_df, target_scaler

#sequences
def prepare_sequences(df, feature_cols):
    X, y = [], []
    for cycle, group in df.groupby("cycle"):
        X.append(group[feature_cols].values)
        y.append(group["capacity"].iloc[-1])  # last capacity per cycle
    X = pad_sequences(X, dtype='float32', padding='post')
    y = np.array(y, dtype='float32')
    return X, y


