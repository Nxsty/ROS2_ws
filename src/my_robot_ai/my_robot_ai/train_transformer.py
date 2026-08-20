import torch
import torch.nn as nn
import h5py
import numpy as np
import os
import glob
from torch.utils.data import Dataset, DataLoader

class CircuitTrajectoryDataset(Dataset):
    def __init__(self, folder_path, seq_length=5):
        self.seq_length = seq_length
        self.lidar_data = []
        self.vel_data = []
        
        files = glob.glob(os.path.join(folder_path, "trajectory_dataset_*.h5"))
        if not files:
            files = glob.glob(os.path.join(folder_path, "*.h5"))
            
        print(f"Cargando {len(files)} archivos para aprendizaje por imitación...")
        
        for file in files:
            with h5py.File(file, 'r') as f:
                self.lidar_data.append(np.array(f['lidar']))
                self.vel_data.append(np.array(f['velocity']))
        
        self.lidar_all = np.concatenate(self.lidar_data, axis=0)
        self.vel_all = np.concatenate(self.vel_data, axis=0)
        
    def __len__(self):
        return len(self.lidar_all) - self.seq_length
    
    def __getitem__(self, idx):
        lidar_seq = self.lidar_all[idx : idx + self.seq_length]
        vel_seq = self.vel_all[idx : idx + self.seq_length]
        target_vel = self.vel_all[idx + self.seq_length]
        return (torch.FloatTensor(lidar_seq), 
                torch.FloatTensor(vel_seq), 
                torch.FloatTensor(target_vel))

class BehavioralTransformer(nn.Module):
    def __init__(self, lidar_dim=360, vel_dim=2, embed_dim=256, nhead=8, layers=4):
        super().__init__()
        
        self.lidar_net = nn.Sequential(
            nn.Linear(lidar_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )
        self.vel_net = nn.Linear(vel_dim, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, 5, embed_dim)) # Seq length = 5
        
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=layers)
        
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, lidar, vel):
        l_feat = self.lidar_net(lidar)
        v_feat = self.vel_net(vel)
        x = l_feat + v_feat + self.pos_emb
        x = self.transformer(x)
        last_state = x[:, -1, :]
        return self.decoder(last_state)

def train_circuit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando en: {device}")
    
    dataset = CircuitTrajectoryDataset("datasets", seq_length=5)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = BehavioralTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    epochs = 200
    for epoch in range(epochs):
        t_loss = 0
        for lidar, vel, target in loader:
            lidar, vel, target = lidar.to(device), vel.to(device), target.to(device)
            optimizer.zero_grad()
            pred = model(lidar, vel)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {t_loss/len(loader):.6f}")
            
    torch.save(model.state_dict(), "datasets/circuit_model.pth")
    print("Modelo optimizado guardado en datasets/circuit_model.pth")

if __name__ == "__main__":
    train_circuit()
