import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import torch
import torch.nn as nn
import numpy as np
import os

class BehavioralTransformer(nn.Module):
    def __init__(self, lidar_dim=360, vel_dim=2, embed_dim=256, nhead=8, layers=4):
        super().__init__()
        self.lidar_net = nn.Sequential(nn.Linear(lidar_dim, 512), nn.ReLU(), nn.Linear(512, embed_dim))
        self.vel_net = nn.Linear(vel_dim, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, 5, embed_dim))
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.decoder = nn.Sequential(nn.Linear(embed_dim, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, lidar, vel):
        l_feat = self.lidar_net(lidar)
        v_feat = self.vel_net(vel)
        x = l_feat + v_feat + self.pos_emb
        x = self.transformer(x)
        return self.decoder(x[:, -1, :])

class CircuitAutopilot(Node):
    def __init__(self):
        super().__init__('circuit_autopilot')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BehavioralTransformer().to(self.device)
        model_path = "datasets/circuit_model.pth"
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.get_logger().info('Modelo OPTIMIZADO cargado correctamente.')
        
        self.lidar_buffer = []
        self.vel_buffer = []
        self.current_vel = [0.0, 0.0]
        
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def odom_callback(self, msg):
        self.current_vel = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.nan_to_num(ranges, nan=12.0, posinf=12.0, neginf=0.0) / 12.0
        if len(ranges) != 360: ranges = np.resize(ranges, (360,))
        
        self.lidar_buffer.append(ranges)
        self.vel_buffer.append(self.current_vel)
        
        if len(self.lidar_buffer) > 5:
            self.lidar_buffer.pop(0)
            self.vel_buffer.pop(0)
            
        if len(self.lidar_buffer) == 5:
            self.predict()

    def predict(self):
        l_in = torch.FloatTensor(np.array([self.lidar_buffer])).to(self.device)
        v_in = torch.FloatTensor(np.array([self.vel_buffer])).to(self.device)
        with torch.no_grad():
            action = self.model(l_in, v_in).cpu().numpy()[0]
        cmd = Twist()
        cmd.linear.x = float(action[0])
        cmd.angular.z = float(action[1])
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = CircuitAutopilot()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
