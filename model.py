from stockdata import df
from numpy import array
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
import torch

scaler = StandardScaler()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df_values = array(df).reshape(-1, 1)
df_scaled = scaler.fit_transform(df_values)

def create_data(data_scaled, sequence_length=30):
    X, y = [], []
    for i in range(len(data_scaled) - sequence_length):
        window = data_scaled[i : i + sequence_length]
        X.append(window)
        target = data_scaled[i + sequence_length]
        y.append(target)
    return array(X), array(y)

X, y = create_data(df_scaled, sequence_length=30)
train_size = int(0.8 * len(X))

X_train = torch.FloatTensor(X[:train_size]).to(device)
y_train = torch.FloatTensor(y[:train_size]).to(device)

X_test = torch.FloatTensor(X[train_size:]).to(device)
y_test = torch.FloatTensor(y[train_size:]).to(device)

class Net(nn.Module):
    def __init__(self, input_layer, hidden_layers, output_layer, num_layers, dropout=0.2):
        super(Net, self).__init__()
        self.num_layers = num_layers
        self.hidden_layers = hidden_layers
        self.lstm = nn.LSTM(input_layer, hidden_layers, num_layers, 
                            batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_layers, output_layer)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

model = Net(1, 128, 1, 2, dropout=0.2).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=50
)

def train(model, X_train, y_train, criterion, optimizer, scheduler, num_epochs):
    for t in range(num_epochs):
        model.train()
        y_prediction = model(X_train)
        loss = criterion(y_prediction, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        scheduler.step(loss.item())
        
    return model

model = train(model, X_train, y_train, criterion, optimizer, scheduler, 1500)

model.eval()
with torch.no_grad():
    y_test_pred = model(X_test)

y_pred = y_test_pred.cpu().numpy()
y_test = y_test.cpu().numpy()

y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE: {rmse:.2f} rubles")