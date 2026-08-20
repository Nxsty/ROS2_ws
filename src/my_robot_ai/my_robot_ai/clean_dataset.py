import h5py
import numpy as np
import os
import glob

def clean_datasets(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.h5"))
    print(f"Limpiando {len(files)} archivos...")
    
    for file in files:
        if "_cleaned" in file: continue
        
        with h5py.File(file, 'r') as f:
            lidar = np.array(f['lidar'])
            pose = np.array(f['pose'])
            vel = np.array(f['velocity'])
            cmd = np.array(f['cmd'])
            ts = np.array(f['timestamp'])
            
        # Encontrar donde el robot se estaba moviendo (vel lineal > 0.05 o angular > 0.05)
        moving_mask = (np.abs(vel[:, 0]) > 0.05) | (np.abs(vel[:, 1]) > 0.05)
        
        # Mantener solo los frames donde hay movimiento
        # (Esto ayuda al modelo a no aprender a "quedarse quieto" por defecto)
        lidar_c = lidar[moving_mask]
        pose_c = pose[moving_mask]
        vel_c = vel[moving_mask]
        cmd_c = cmd[moving_mask]
        ts_c = ts[moving_mask]
        
        if len(lidar_c) < 100:
            print(f"Archivo {file} tiene muy pocos datos de movimiento, saltando...")
            continue
            
        new_file = file.replace(".h5", "_cleaned.h5")
        with h5py.File(new_file, 'w') as f:
            f.create_dataset('lidar', data=lidar_c)
            f.create_dataset('pose', data=pose_c)
            f.create_dataset('velocity', data=vel_c)
            f.create_dataset('cmd', data=cmd_c)
            f.create_dataset('timestamp', data=ts_c)
            
        print(f"Creado {new_file} con {len(lidar_c)} muestras (antes {len(lidar)})")

if __name__ == "__main__":
    clean_datasets("datasets")
