import torch
import torch.nn as nn
import h5py
import numpy as np
import os
import glob
from torch.utils.data import Dataset, DataLoader

class GoalTrajectoryDataset(Dataset):
    def __init__(self, folder_path, seq_length=10, horizon=20):
        self.seq_length = seq_length
        self.horizon = horizon
        self.lidar_data = []
        self.goal_data = []
        self.pose_data = []
        self.vel_data = []
        
        # Cargar archivos manuales avanzados y los de metas automáticas
        files = glob.glob(os.path.join(folder_path, "advanced_manual_*.h5")) + \
                glob.glob(os.path.join(folder_path, "goal_dataset_*.h5"))
        print(f"Cargando {len(files)} archivos de entrenamiento...")
        
        for file in files:
            with h5py.File(file, 'r') as f:
                if 'goal' not in f: continue
                self.lidar_data.append(np.array(f['lidar']))
                self.goal_data.append(np.array(f['goal']))
                self.pose_data.append(np.array(f['pose']))
                self.vel_data.append(np.array(f['velocity']))
        
        self.lidar_all = np.concatenate(self.lidar_data, axis=0) / 12.0
        self.goal_all = np.concatenate(self.goal_data, axis=0)
        self.pose_all = np.concatenate(self.pose_data, axis=0)
        self.vel_all = np.concatenate(self.vel_data, axis=0)
        
    def __len__(self):
        return len(self.lidar_all) - self.seq_length - self.horizon
    
    def __getitem__(self, idx):
        lidar_seq = self.lidar_all[idx : idx + self.seq_length]
        # La meta relativa (distancia, angulo) en el momento actual
        goal_current = self.goal_all[idx + self.seq_length - 1]
        
        # Target: Próximos movimientos relativos
        curr_pose = self.pose_all[idx + self.seq_length - 1]
        future_poses = self.pose_all[idx + self.seq_length : idx + self.seq_length + self.horizon]
        
        rel_traj = []
        cx, cy, cyaw = curr_pose
        for fx, fy, fyaw in future_poses:
            dx, dy = fx - cx, fy - cy
            lx = dx * np.cos(-cyaw) - dy * np.sin(-cyaw)
            ly = dx * np.sin(-cyaw) + dy * np.cos(-cyaw)
            rel_traj.append([lx, ly]) # Solo x, y para simplificar
            
        return (torch.FloatTensor(lidar_seq), 
                torch.FloatTensor(goal_current), 
                torch.FloatTensor(rel_traj))

class GoalTransformer(nn.Module):
    def __init__(self, lidar_dim=360, goal_dim=2, embed_dim=128, horizon=20):
        super().__init__()
        self.lidar_enc = nn.Linear(lidar_dim, embed_dim)
        self.goal_enc = nn.Linear(goal_dim, embed_dim)
        
        self.pos_emb = nn.Parameter(torch.randn(1, 10, embed_dim))
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=3)
        
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, horizon * 2)
        )
        self.horizon = horizon

    def forward(self, lidar, goal):
        # lidar: [B, 10, 360], goal: [B, 2]
        l_feat = self.lidar_enc(lidar) # [B, 10, E]
        g_feat = self.goal_enc(goal).unsqueeze(1) # [B, 1, E]
        
        # Concatenamos la meta como un "token" extra de la secuencia
        x = torch.cat([g_feat, l_feat], dim=1) # [B, 11, E]
        x = self.transformer(x)
        
        # Usamos el token de la meta procesado para predecir
        out = self.decoder(x[:, 0, :])
        return out.view(-1, self.horizon, 2)

def train_goal_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = GoalTrajectoryDataset("datasets")
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = GoalTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    print(f"Entrenando Transformer Condicionado por Meta en {device}...")
    for epoch in range(100):
        t_loss = 0
        for lidar, goal, target in loader:
            lidar, goal, target = lidar.to(device), goal.to(device), target.to(device)
            optimizer.zero_grad()
            pred = model(lidar, goal)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/100, Loss: {t_loss/len(loader):.6f}")
            
    torch.save(model.state_dict(), "datasets/goal_transformer.pth")
    print("Modelo de Metas guardado.")

if __name__ == "__main__":
    train_goal_model()
