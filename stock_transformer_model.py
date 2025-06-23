import torch
import torch.nn as nn

class StockTransformer(nn.Module):
    def __init__(self, input_dim=5, seq_len=10, d_model=64, nhead=4, num_layers=2, num_classes=3):
        super(StockTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * seq_len, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.classifier(x)


# trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

class Trainer:
    def __init__(self, model, X, y, batch_size=32, lr=1e-3, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.batch_size = batch_size
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loader = DataLoader(TensorDataset(self.X, self.y), batch_size=self.batch_size, shuffle=True)

    def train(self, epochs=10):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for xb, yb in self.loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(self.X.to(self.device)).argmax(dim=1).cpu().numpy()
            print(classification_report(self.y, preds, target_names=["Down", "Neutral", "Up"]))
