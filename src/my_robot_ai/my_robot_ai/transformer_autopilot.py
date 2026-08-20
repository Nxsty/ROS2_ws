import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import torch
import torch.nn as nn
import numpy as np
import os
from math import atan2, asin

# Re-importing architecture (must match train_transformer.py)
class TrajectoryTransformer(nn.Module):
    def __init__(self, lidar_dim=360, vel_dim=2, embed_dim=128, nhead=8, layers=4, horizon=20):
        super().__init__()
        self.lidar_net = nn.Sequential(nn.Linear(lidar_dim, 256), nn.ReLU(), nn.Linear(256, embed_dim))
        self.vel_net = nn.Linear(vel_dim, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, 10, embed_dim))
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.decoder = nn.Sequential(nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, horizon * 3))
        self.horizon = horizon

    def forward(self, lidar, vel):
        l_feat = self.lidar_net(lidar)
        v_feat = self.vel_net(vel)
        x = l_feat + v_feat + self.pos_emb
        x = self.transformer(x)
        last_state = x[:, -1, :]
        out = self.decoder(last_state)
        return out.view(-1, self.horizon, 3)

class TransformerAutopilot(Node):
    def __init__(self):
        super().__init__('transformer_autopilot')
        
        # Load Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TrajectoryTransformer().to(self.device)
        model_path = "datasets/trajectory_transformer.pth"
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.get_logger().info(f'Modelo cargado exitosamente de {model_path}')
        else:
            self.get_logger().error(f'No se encontró el modelo en {model_path}')
            return

        # Buffers for sequence [Seq=10]
        self.lidar_buffer = []
        self.vel_buffer = []
        
        # Subs and Pubs
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.current_vel = [0.0, 0.0]
        self.get_logger().info('Autopiloto Transformer iniciado. Esperando datos...')

    def scan_callback(self, msg):
        # 1. Preprocess LIDAR
        ranges = np.array(msg.ranges)
        ranges = np.nan_to_num(ranges, nan=12.0, posinf=12.0, neginf=0.0)
        if len(ranges) != 360:
            ranges = np.resize(ranges, (360,))
        ranges = ranges / 12.0 # Normalization
        
        # 2. Update Buffers
        self.lidar_buffer.append(ranges)
        self.vel_buffer.append(self.current_vel)
        
        if len(self.lidar_buffer) > 10:
            self.lidar_buffer.pop(0)
            self.vel_buffer.pop(0)
            
        # 3. Predict if buffer is full
        if len(self.lidar_buffer) == 10:
            self.predict_and_control()

    def odom_callback(self, msg):
        self.current_vel = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]

    def predict_and_control(self):
        # Convert to tensor
        lidar_in = torch.FloatTensor(np.array([self.lidar_buffer])).to(self.device)
        vel_in = torch.FloatTensor(np.array([self.vel_buffer])).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(lidar_in, vel_in)
            prediction = prediction.cpu().numpy()[0] # [Horizon=20, 3]
            
        # prediction[i] = [lx, ly, lyaw] relative to robot
        # Simple control: look at a point in the near future (e.g. index 5 of 20)
        target_idx = 5
        target_x = prediction[target_idx][0]
        target_y = prediction[target_idx][1]
        
        # Calculate steering to reach target_x, target_y
        angle_to_target = atan2(target_y, target_x)
        
        cmd = Twist()
        # Linear velocity proportional to distance (but capped)
        dist = np.sqrt(target_x**2 + target_y**2)
        cmd.linear.x = min(0.3, dist * 1.0) 
        
        # Angular velocity proportional to angle
        cmd.angular.z = angle_to_target * 2.0
        
        # Safety check: if very close to obstacle in front, stop
        # (Lidar points 170-190 are roughly front)
        front_ranges = self.lidar_buffer[-1][170:190]
        if np.min(front_ranges) * 12.0 < 0.4:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5 # Turn to find path
            
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = TransformerAutopilot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
