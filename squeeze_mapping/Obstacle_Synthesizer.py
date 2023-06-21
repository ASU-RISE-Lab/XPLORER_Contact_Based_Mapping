import sys
import os
import time
import math
import rclpy 
from rclpy.node import Node
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs

import random

import numpy as np
import open3d as o3d

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial.transform import Rotation

from squeeze_custom_msgs.msg import ImuData
from squeeze_custom_msgs.msg import ExternalWrenchEstimate

from std_msgs.msg import Float64

from px4_msgs.msg import VehicleAttitude
from px4_msgs.msg import SensorCombined
from px4_msgs.msg import ControllerOut
from px4_msgs.msg import VehicleOdometry
from px4_msgs.msg import VehicleControlMode
from px4_msgs.msg import VehicleStatus

class PCDStorage(Node):

    def __init__(self):
        super().__init__('pcd_publisher_node')

        self.initialize_variables()

        print("Obstacle Synthesizer Node Started")
        self.start_time = time.time()

        # Subscribers

        self.imu_angle_sub = self.create_subscription(ImuData, 'Arm_Angles_Filtered', self.imu_angle_callback, 1)
        self.vehicle_odometry_sub = self.create_subscription(VehicleOdometry, '/fmu/vehicle_odometry/out', self.vehicle_odometry_callback, 1)
        self.vehicle_control_mode_sub = self.create_subscription(VehicleControlMode, '/fmu/vehicle_control_mode/out', self.vehicle_control_mode_callback, 1)
        self.wrench_sub = self.create_subscription(ExternalWrenchEstimate, '/External_Wrench_Estimate_Avg', self.wrench_callback, 1)
        self.move_direction_sub = self.create_subscription(Float64, '/Move_Direction', self.move_direction_callback, 1)
        self.yaw_gen_state_sub = self.create_subscription(Float64, "/Yaw_Generate_State", self.yaw_gen_state_callback, 1)
        self.vehicle_status_sub = self.create_subscription(VehicleStatus, "/fmu/vehicle_status/out", self.vehicle_status_callback, 1)

        self.ply_reader()   # Read the PLY files

        # fname = "Final_Maps/Wall_Traversal_Master" + ".ply" 
        fname = "Final_Maps/Box_Traversal_Full" + ".ply"
        self.outfile = "/home/"+os.environ.get("USERNAME")+"/colcon_ws_squeeze/src/squeeze_mapping/maps/Final_Maps/" + fname

        self.pcd_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'pcd', 10)
        timer_period = 1/30.0 # 30 For better Frame Rate
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # self.visualize()
        # self.timer = self.create_timer(timer_period, self.visualize_callback)

    def initialize_variables(self):
        """
        Wall Tracing : 
        self.x = 0.241285
        self.y = 0.694175
        self.z = 0.150749
        """
        """
        3 - Sided Box Tracing :
        self.x = -0.592219
        self.y = 0.240052
        self.z = 0.161033
        """
        """
        Complex Box Tracing :
        self.x = -0.95
        self.y = 0.0
        self.z = 0.7
        """
        self.x = -0.95
        self.y = 0.0
        self.z = 0.7
        self.yaw = 0.0
        self.q = [0,0,0,0]

        self.pts = np.array([[0,0,0]])

        self.f_x = 0.0
        self.f_y = 0.0
        self.f_z = 0.0

        self.move_direction = 2
        self.move_direction_prev = 0

        self.offboard_status = False
        self.armed_status = False
        self.map_flag = False
        self.buffer_count = 0
        self.points_buffer = np.zeros((1,3))

    def visualize(self):
        view_pts = o3d.io.read_point_cloud(self.outfile)

        self.view_pts = np.asarray(view_pts.points)



        self.pcd_publisher_func(self.view_pts)

        # o3d.visualization.draw_geometries([view_pts])

    def visualize_callback(self):
        self.pcd_publisher_func(self.view_pts)
        print("Map Visualized - ", time.time())

    def timer_callback(self):
        
        if(abs(self.f_x) > 1.51 or abs(self.f_y) > 1.51):  # Box
        # if(abs(self.f_x) > 1.8 or abs(self.f_y) > 1.8):      # Wall
            self.map_flag = True

        if(abs(self.f_x) < 0.195 and abs(self.f_y) < 0.195 and self.map_flag == True):
            self.map_flag = False

        if (self.map_flag == True and self.offboard_status == True):

            theta, offset = self.wall_direction()
            R1 = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, theta])
            pts = self.wall_pts @ R1
            pts = pts + offset
            R2 = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, self.yaw])
            pts = pts @ R2
            pts = pts + np.array([self.x, self.y, 0.0])

            self.store_pts(pts)

        # if (self.map_flag == True and self.armed_status == False):
        #     self.write_to_file()

            self.pcd_publisher_func(pts)

    def wall_direction(self):

        if(self.move_direction == 0):
            return -math.pi/2, np.array([0.0,+0.21,0.0])
        elif(self.move_direction == 1):
            # return math.pi/2, np.array([0.21,-0.27,0.0])
            return math.pi/2, np.array([0.0,-0.21,0.0])
        elif(self.move_direction == 2):
            return 0.0, np.array([+0.18,0.0,0.0]) # Box
            # return 0.0, np.array([+0.21,-0.03,0.0]) # Wall  
        elif(self.move_direction == 3):
            return math.pi, np.array([-0.21,0.0,0.0])

    def store_pts(self,points):

        if(self.offboard_status):
            self.points_buffer = np.append(self.points_buffer, points, axis=0)
            print("Points Stored - ", time.time() - self.start_time)

    def write_to_file(self):

            pts = np.asarray(self.points_buffer)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            # pcd = pcd.uniform_down_sample(every_k_points=1)
            down_pcd = pcd.voxel_down_sample(voxel_size=0.01)
            o3d.io.write_point_cloud("/home/napster/colcon_ws_squeeze/src/squeeze_mapping/maps/Final_Maps/Wall_Traversal_Master.ply"
                                     ,down_pcd, write_ascii=True, compressed=False, print_progress=True)
            print("PC File Created - ",time.time())
            exit()

    def ply_reader(self):

        wall_path = "/home/napster/colcon_ws_squeeze/src/squeeze_mapping/maps/Final_Maps/New_Wall_in_m.ply"

        self.corner_block_path = "/home/napster/colcon_ws_squeeze/src/squeeze_mapping/maps/Final_Maps/Corner_Block_in_m.ply"

        wall_pcd = o3d.io.read_point_cloud(wall_path)

        # wall_pcd.transform([[1,0,0,0],[0,1,0,0],[0,0,1,0.085],[0,0,0,1]])

        self.wall_pts = np.asarray(wall_pcd.points)

        R = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, np.pi/2]) # To orient about X-Axis
        self.wall_pts = self.wall_pts @ R
        self.wall_pts = self.wall_pts + np.array([0.0,0.05,0.0]) # To shift the wall to the center of the origin axes

        # pts = self.wall_pts

        # theta, offset = self.wall_direction()
        # R1 = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, theta])
        # pts = self.wall_pts @ R1
        # pts = pts + offset
        # coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pts)
        # o3d.visualization.draw_geometries([pcd,coord])

        # print("Wall Points : ",self.wall_pts.shape)

    def pcd_publisher_func(self,points):

        pcd = point_cloud(points, 'map')
        self.pcd_publisher.publish(pcd)

    def vehicle_control_mode_callback(self, msg):
        self.armed_status = msg.flag_armed
        # print("Armed Status : ",self.armed_status)
        
    def imu_angle_callback(self, msg):
        
        """
        3       2
         -     -
           -  -
            -  
          -   -  
        4       1
        """

        self.arm_1_angle = msg.imu1 * np.pi/180
        self.arm_2_angle = msg.imu2 * np.pi/180
        self.arm_3_angle = msg.imu3 * np.pi/180
        self.arm_4_angle = msg.imu4 * np.pi/180

    def vehicle_odometry_callback(self,msg):
        self.timestamp = msg.timestamp
        self.x = round(msg.x,4)
        self.y = -round(msg.y,4)  # To Convert from NED(X North, Y East, Z Down) to FLU(X Forward, Y Left, Z Up) Frame 
        self.z = -round(msg.z,4)

        self.q[3] = msg.q[0]
        self.q[0] = msg.q[1]
        self.q[1] = msg.q[2]
        self.q[2] = msg.q[3]

        self.rot = Rotation.from_quat(self.q)
        self.rot_euler = self.rot.as_euler('xyz', degrees=True)  # [roll, pitch, yaw]
        self.yaw = (self.rot_euler[2] * math.pi / 180.0)  # To Convert from NED(X North, Y East, Z Down) to FLU(X Forward, Y Left, Z Up) Frame

        self.yaw_speed = msg.yawspeed

    def wrench_callback(self, msg):

        try:
            self.f_x = msg.f_x
            self.f_y = msg.f_y
            self.f_z = msg.f_z

            self.tau_p = msg.tau_p
            self.tau_q = msg.tau_q
            self.tau_r = msg.tau_r
    
        except Exception as e:
            print("Exception in wrench_callback : ",e)

    def move_direction_callback(self, msg):
        self.move_direction = msg.data

        if self.move_direction == 1.0 and self.move_direction_prev == 2.0 and self.map_flag == True : # Right Top Corner

            pts = o3d.io.read_point_cloud(self.corner_block_path)
            pts = np.asarray(pts.points)
            pts = pts + np.array([0.295,-0.315,0.03]) + np.array([self.x,self.y,0.0])
            
            self.store_pts(pts)
            self.pcd_publisher_func(pts)
            print("Corner Block Published - ",time.time())

        self.move_direction_prev = self.move_direction

    def yaw_gen_state_callback(self,msg):
        self.yaw_gen_state = msg.data

    def vehicle_status_callback(self, msg):

        if msg.nav_state == 14:
            self.offboard_status = True
        else:
            self.offboard_status = False

def point_cloud(points, parent_frame):
    """ Creates a point cloud message.
    Args:
        points: Nx3 array of xyz positions.
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message

    Code source:
        https://gist.github.com/pgorczak/5c717baa44479fa064eb8d33ea4587e0

    References:
        http://docs.ros.org/melodic/api/sensor_msgs/html/msg/PointCloud2.html
        http://docs.ros.org/melodic/api/sensor_msgs/html/msg/PointField.html
        http://docs.ros.org/melodic/api/std_msgs/html/msg/Header.html

    """
    # In a PointCloud2 message, the point cloud is stored as an byte 
    # array. In order to unpack it, we also include some parameters 
    # which desribes the size of each individual point.
    ros_dtype = sensor_msgs.PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize # A 32-bit float takes 4 bytes.

    data = points.astype(dtype).tobytes() 

    # The fields specify what the bytes represents. The first 4 bytes 
    # represents the x-coordinate, the next 4 the y-coordinate, etc.
    fields = [sensor_msgs.PointField(
        name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate('xyz')]

    # The PointCloud2 message also has a header which specifies which 
    # coordinate frame it is represented in. 
    header = std_msgs.Header(frame_id=parent_frame)

    return sensor_msgs.PointCloud2(
        header=header,
        height=1, 
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 3), # Every point consists of three float32s.
        row_step=(itemsize * 3 * points.shape[0]),
        data=data
    )


def main(args=None):
    # Boilerplate code.
    rclpy.init(args=args)
    pcd_publisher = PCDStorage()
    rclpy.spin(pcd_publisher)
    
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    pcd_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
