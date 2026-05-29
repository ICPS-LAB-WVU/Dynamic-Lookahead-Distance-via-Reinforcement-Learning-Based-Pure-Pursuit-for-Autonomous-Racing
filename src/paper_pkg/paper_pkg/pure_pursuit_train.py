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
        self.map_name = 'Austin_fast'
        self.is_ascending = True
        self.prev_steering = 0.0  # (unused now; kept for easy rollback)

        # Topics
        drive_topic = '/drive'
        odom_topic = '/pf/viz/inferred_pose' if self.is_real else '/ego_racecar/odom'
        visualization_topic = '/visualization_marker_array'

        # Publishers & Subscribers
        self.sub_pose = self.create_subscription(
            PoseStamped if self.is_real else Odometry, odom_topic, self.pose_callback, 1
        )
        self.pub_drive = self.create_publisher(AckermannDriveStamped, drive_topic, 1)
        self.pub_vis = self.create_publisher(MarkerArray, visualization_topic, 1)

        # (Optional) still subscribed; not used for control in this variant
        self.sub_ld = self.create_subscription(Float32, '/lookahead_distance', self.lookahead_callback, 10)
        self.sub_reset = self.create_subscription(Float32, '/episode_reset_signal', self.reset_callback, 10)

        self.drive_msg = AckermannDriveStamped()
        self.markerArray = MarkerArray()

        # Load waypoints (expects ; delimiter and header lines)
        map_path = os.path.abspath(os.path.join('src/paper_ws/src', 'csv_data'))
        csv_data = np.loadtxt(f'{map_path}/{self.map_name}.csv', delimiter=';', skiprows=3)
        self.waypoints = csv_data[:, 1:3]          # x, y
        self.ref_speed = csv_data[:, 5] * 1.5      # speed profile (scaled)
        self.curvatures = csv_data[:, 4]           # kappa_radpm (not used in this variant)
        self.numWaypoints = self.waypoints.shape[0]

        # Defaults (will be overwritten each tick)
        self.L = 1.0
        self.steering_gain = 0.5

        self.visualization_init()

    # --- kept for compatibility; not used by control logic in this variant ---
    def lookahead_callback(self, msg):
        self.L = msg.data

    def reset_callback(self, msg):
        self.current_idx = 0
        self.flag = False

    # === NEW: map speed -> (lookahead, steering_gain) ===
    def build_control_functions(self, speed, v_min, v_max):
        """
        Lookahead mapping: v_min -> d_min, v_max -> d_max
        Steering gain mapping decreases with speed.
        """
        # Lookahead distance mapping (your numbers/comment)
        d_min = 1.0
        d_max = 2.5
        m_L = (d_max - d_min) / (v_max - v_min)
        b_L = d_min - m_L * v_min
        L = m_L * speed + b_L

        # Steering gain mapping (your numbers)
        gain_max = 0.9
        gain_min = 0.65
        m_gain = (gain_min - gain_max) / (v_max - v_min)
        b_gain = gain_max - m_gain * v_min
        steering_gain = m_gain * speed + b_gain

        return float(L), float(steering_gain)

    def pose_callback(self, pose_msg):
        # --- pose/rotation ---
        if self.is_real:
            pos = pose_msg.pose
            quat = pose_msg.pose.orientation
        else:
            pos = pose_msg.pose.pose
            quat = pose_msg.pose.pose.orientation

        self.currX = pos.position.x
        self.currY = pos.position.y
        self.currPos = np.array([self.currX, self.currY]).reshape(1, 2)

        Rq = transform.Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
        self.rot = Rq.as_matrix()

        # --- waypoint indexing ---
        self.distances = distance.cdist(self.currPos, self.waypoints, 'euclidean').reshape(self.numWaypoints)
        self.closest_index = int(np.argmin(self.distances))
        self.closestPoint = self.waypoints[self.closest_index]

        # --- reference speed from raceline ---
        speed = float(self.ref_speed[self.closest_index])

        # --- speed → (L, gain) ---
        v_min, v_max = 3.0, 18.0
        self.L, self.steering_gain = self.build_control_functions(speed, v_min, v_max)

        # Optional extra lookahead at higher speed (as requested)
        #if speed >= 6.3:
         #   self.L += 1.0

        # --- target point and transform to vehicle frame ---
        targetPoint = self.get_closest_point_beyond_lookahead_dist(self.L)
        translatedTargetPoint = self.translatePoint(targetPoint)

        # --- curvature/steering ---
        y = translatedTargetPoint[1]
        gamma = self.steering_gain * (2.0 * y / (self.L ** 2))
        gamma = float(np.clip(gamma, -0.35, 0.35))

        # --- publish drive (cap real-car speed for safety) ---
        self.drive_msg.drive.steering_angle = gamma
        self.drive_msg.drive.speed = speed if not self.is_real else min(speed, 2.0)
        self.pub_drive.publish(self.drive_msg)

        print(f"Pos=({self.currX:.2f}, {self.currY:.2f}), Steer={gamma:.3f}, Speed={self.drive_msg.drive.speed:.2f}, L={self.L:.2f}, gain={self.steering_gain:.2f}")

        # --- visualize ---
        self.targetMarker.points = [Point(x=targetPoint[0], y=targetPoint[1], z=0.0)]
        self.closestMarker.points = [Point(x=self.closestPoint[0], y=self.closestPoint[1], z=0.0)]
        self.markerArray.markers = [self.waypointMarker, self.targetMarker, self.closestMarker]
        self.pub_vis.publish(self.markerArray)

    def get_closest_point_beyond_lookahead_dist(self, threshold):
        """Advance along waypoints in the chosen direction until distance >= threshold."""
        idx = int(self.closest_index)
        if self.is_ascending:
            while self.distances[idx] < threshold:
                idx = (idx + 1) % self.numWaypoints
        else:
            while self.distances[idx] < threshold:
                idx = (idx - 1 + self.numWaypoints) % self.numWaypoints
        return self.waypoints[idx]

    def translatePoint(self, point):
        """Express the world point in the vehicle frame (x forward, y left)."""
        pv = (point - self.currPos).reshape(2)
        v_local = self.rot.T @ np.array([pv[0], pv[1], 0.0])
        return v_local  # [x_local, y_local, z_local]

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
