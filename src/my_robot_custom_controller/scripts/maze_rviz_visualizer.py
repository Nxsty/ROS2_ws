#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
import os

class MazeVisualizer(Node):
    def __init__(self):
        super().__init__('maze_visualizer')
        self.publisher = self.create_publisher(Marker, 'maze_marker', 10)
        self.timer = self.create_timer(1.0, self.publish_maze)
        
        # Ruta al archivo STL (ajustada a tu estructura)
        self.stl_path = 'package://my_robot_normal_bringup/mesh/maze.stl'
        
        self.get_logger().info('Visualizador de Laberinto para RViz iniciado.')

    def publish_maze(self):
        marker = Marker()
        marker.header.frame_id = "odom" 
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "maze"
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        
        # --- ALINEACIÓN CON GAZEBO ---
        # Robot spawn: x=-0.8116, y=0.9690
        # Para que el origen de Gazebo coincida en el frame 'odom':
        marker.pose.position.x = 0.8116
        marker.pose.position.y = -0.9690
        marker.pose.position.z = -0.5 # Bajado a -0.5 para tocar el suelo
        marker.pose.orientation.w = 1.0
        
        # Escala
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        
        # Intentamos usar la ruta absoluta para evitar problemas de resolución
        marker.mesh_resource = "package://my_robot_normal_bringup/mesh/maze.stl"
        marker.mesh_use_embedded_materials = True
        
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0 # Opaco para que se vea bien
        
        marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg() # Infinito
        
        self.publisher.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = MazeVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
