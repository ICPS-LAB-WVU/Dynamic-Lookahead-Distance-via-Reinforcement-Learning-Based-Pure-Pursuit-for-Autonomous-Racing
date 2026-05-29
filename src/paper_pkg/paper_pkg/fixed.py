#pure_pursuit.py

#!/usr/bin/env python3

import numpy as np
from scipy.spatial import distance, transform
import os

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import Float32


class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')

        self.is_real = False
        self.map_name = 'YasMarina_fast'
        self.is_ascending = True
        self.prev_steering = 0.0  # for smoothing
        self.speed = 0.0

        # Topics
        drive_topic = '/drive'
        odom_topic = '/pf/viz/inferred_pose' if self.is_real else '/ego_racecar/odom'
        visualization_topic = '/visualization_marker_array'

        # Publishers & Subscribers
        self.sub_pose = self.create_subscription(PoseStamped if self.is_real else Odometry, odom_topic, self.pose_callback, 1)
        self.pub_drive = self.create_publisher(AckermannDriveStamped, drive_topic, 1)
        self.pub_vis = self.create_publisher(MarkerArray, visualization_topic, 1)

        self.sub_ld = self.create_subscription(Float32, '/lookahead_distance', self.lookahead_callback, 10)
        self.sub_reset = self.create_subscription(Float32, '/episode_reset_signal', self.reset_callback, 10)

        self.drive_msg = AckermannDriveStamped()
        self.markerArray = MarkerArray()

        # Load waypoints
        map_path = os.path.abspath(os.path.join('src/paper_ws/src', 'csv_data'))
        csv_data = np.loadtxt(f'{map_path}/{self.map_name}.csv', delimiter=';', skiprows=3)
        self.waypoints = csv_data[:, 1:3]
        self.ref_speed = csv_data[:, 5] * 1.0 
        self.curvatures = csv_data[:, 4]  # kappa_radpm
        self.numWaypoints = self.waypoints.shape[0]

        self.L = 1.2  # Default lookahead
        self.steering_gain = 0.5
        self.visualization_init()

    def lookahead_callback(self, msg):
        self.L = msg.data

    def reset_callback(self, msg):
        self.current_idx = 0
        self.flag = False

    def build_control_gain(self, speed, v_min=2.98, v_max=12.0, gain_min=0.6, gain_max=0.65):
        m = (gain_min - gain_max) / (v_max - v_min)
        b = gain_max - m * v_min
        return max(min(m * speed + b, gain_max), gain_min)

    def pose_callback(self, pose_msg):
        # Get position
        pos = pose_msg.pose if self.is_real else pose_msg.pose.pose
        self.speed = float(pose_msg.twist.twist.linear.x)
        self.currX = pos.position.x
        self.currY = pos.position.y
        self.currPos = np.array([[self.currX, self.currY]])

        # Get rotation matrix
        quat = pos.orientation
        R = transform.Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
        self.rot = R.as_matrix()

        # Find closest waypoint
        self.distances = distance.cdist(self.currPos, self.waypoints).reshape((self.numWaypoints))
        self.closest_index = int(np.argmin(self.distances))
        self.closestPoint = self.waypoints[self.closest_index]
        speed = self.ref_speed[self.closest_index]

        # Store current curvature
        self.curvature = self.curvatures[self.closest_index]

        speed_cmd = float(self.ref_speed[self.closest_index])

        # Update steering gain and lookahead
        self.steering_gain = self.build_control_gain(speed)
        L_dynamic = self.L 

        # Get target point and transform
        target = self.get_lookahead_point(L_dynamic)
        translated = self.transform_to_vehicle_frame(target)

        # Compute curvature and steering angle
        y = translated[1]
        gamma = self.steering_gain * (2 * y / L_dynamic**2)
        gamma = np.clip(gamma, -0.35, 0.35)

        # Smooth steering using a low-pass filter
        alpha = 0.4
        gamma = (1 - alpha) * self.prev_steering + alpha * gamma
        self.prev_steering = gamma

        # Publish drive command
        self.drive_msg.drive.steering_angle = gamma
        self.drive_msg.drive.speed = speed if not self.is_real else min(speed, 2.0)
        self.pub_drive.publish(self.drive_msg)

        print(f"Pos=({self.currX:.2f}, {self.currY:.2f}), Steer={gamma:.3f}, Speed={self.drive_msg.drive.speed:.2f}, v_odom={self.speed:.2f},v_csv={speed_cmd:.2f}")

        # Visualize
        self.targetMarker.points = [Point(x=target[0], y=target[1], z=0.0)]
        self.closestMarker.points = [Point(x=self.closestPoint[0], y=self.closestPoint[1], z=0.0)]
        self.markerArray.markers = [self.waypointMarker, self.targetMarker, self.closestMarker]
        self.pub_vis.publish(self.markerArray)

    def get_lookahead_point(self, threshold):
        idx = self.closest_index
        while self.distances[idx] < threshold:
            idx = (idx + 1) % self.numWaypoints if self.is_ascending else (idx - 1 + self.numWaypoints) % self.numWaypoints
        return self.waypoints[idx]

    def transform_to_vehicle_frame(self, point):
        pvect = point - self.currPos
        H = np.eye(4)
        H[:3, :3] = np.linalg.inv(self.rot)
        H[0, 3] = self.currX
        H[1, 3] = self.currY
        return (H @ np.array([pvect[0, 0], pvect[0, 1], 0, 0])).reshape((4))

    def visualization_init(self):
        def create_marker(marker_id, color, scale):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.type = Marker.POINTS
            marker.id = marker_id
            marker.scale.x = scale
            marker.scale.y = scale
            marker.color.a = 1.0
            setattr(marker.color, color, 0.75)
            return marker

        self.waypointMarker = create_marker(0, 'g', 0.05)
        self.waypointMarker.points = [Point(x=w[0], y=w[1], z=0.0) for w in self.waypoints]
        self.targetMarker = create_marker(1, 'r', 0.2)
        self.closestMarker = create_marker(2, 'b', 0.2)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()