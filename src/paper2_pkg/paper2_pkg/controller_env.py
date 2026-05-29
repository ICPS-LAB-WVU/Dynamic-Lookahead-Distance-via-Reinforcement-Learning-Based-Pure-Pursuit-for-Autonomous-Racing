#!/usr/bin/env python3

import os
import time
import random
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from tf_transformations import quaternion_from_euler
import gym
from gym import spaces

class ControllerArbiterEnv(gym.Env):
    def __init__(self):
        if not rclpy.ok():
            rclpy.init()

        super().__init__()
        self.node = rclpy.create_node('controller_arbiter_env')

        # Load waypoints
        map_path = os.path.abspath(os.path.join('src/paper_ws/src', 'csv_data'))
        csv_data = np.loadtxt(f'{map_path}/final.csv', delimiter=',', skiprows=0)
        self.waypoints = csv_data[:, 1:3]
        self.ref_speed = csv_data[:, 5]
        self.curvatures = csv_data[:, 4]  # kappa_radpm
        self.numWaypoints = self.waypoints.shape[0]

        # Subscriptions
        self.sub_pure_pursuit = self.node.create_subscription(
            AckermannDriveStamped, '/pure_cmd', self.pure_pursuit_callback, 10)
        self.sub_gap_follow = self.node.create_subscription(
            AckermannDriveStamped, '/gap_follow_cmd', self.gap_follow_callback, 10)
        self.sub_lidar = self.node.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10)
        self.sub_pose = self.node.create_subscription(
            PoseStamped, '/ego_racecar/pose', self.pose_callback, 10)

        # Publisher
        self.pub_drive = self.node.create_publisher(AckermannDriveStamped, '/drive', 10)

        # State variables
        self.pure_cmd = None
        self.gap_cmd = None
        self.lidar_ranges = []
        self.collision = False
        self.speed = 0.0
        self.step_count = 0
        self.max_steps = 1000

        # Reset variables
        self.stalled_steps = 0
        self.stall_limit = 200
        self.current_wp_idx = 0
        self.last_wp_idx = 0

    def pure_pursuit_callback(self, msg):
        self.pure_cmd = msg
        self.speed = msg.drive.speed

    def gap_follow_callback(self, msg):
        self.gap_cmd = msg

    def lidar_callback(self, msg):
        self.lidar_ranges = msg.ranges
        if len(msg.ranges) > 0 and min(msg.ranges) < 0.2:
            self.collision = True

    def pose_callback(self, msg):
        if self.waypoints is None or len(self.waypoints) == 0:
            return
        car_pos = np.array([msg.pose.position.x, msg.pose.position.y])
        dists = np.linalg.norm(self.waypoints - car_pos, axis=1)
        closest_idx = np.argmin(dists)
        if closest_idx > self.current_wp_idx:
            self.current_wp_idx = closest_idx

    def reset_car_position(self):
        reset_pub = self.node.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'

        idx = random.randint(0, len(self.waypoints) - 2)
        x, y = self.waypoints[idx]
        next_x, next_y = self.waypoints[idx + 1]
        dx, dy = next_x - x, next_y - y
        theta = np.arctan2(dy, dx)

        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        q = quaternion_from_euler(0, 0, theta)
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]

        reset_pub.publish(msg)
        time.sleep(0.5)

    def reset(self):
        self.collision = False
        self.step_count = 0
        self.stalled_steps = 0
        self.current_wp_idx = 0
        self.last_wp_idx = 0

        self.reset_car_position()

        rclpy.spin_once(self.node, timeout_sec=0.1)

        return self._get_obs()

    def step(self, action):
        self.step_count += 1

        # Action: 0 means pure pursuit, 1 means gap follow
        if action == 0 and self.pure_cmd is not None:
            cmd = self.pure_cmd
        elif action == 1 and self.gap_cmd is not None:
            cmd = self.gap_cmd
        else:
            cmd = AckermannDriveStamped()
            cmd.drive.speed = 0.0
            cmd.drive.steering_angle = 0.0

        self.pub_drive.publish(cmd)

        rclpy.spin_once(self.node, timeout_sec=0.05)

        reward = 0.0
        done = False

        if self.collision:
            reward -= 20.0
            done = True
            self.reset_car_position()
        else:
            # Positive reward for progress
            progress = self.current_wp_idx - self.last_wp_idx
            if progress > 0:
                reward += progress * 5.0  # reward for waypoint progress
                self.last_wp_idx = self.current_wp_idx
            # Reward for moving forward speed (small positive)
            reward += 0.1 * self.speed

        if self.step_count >= self.max_steps:
            done = True

        obs = self._get_obs()
        info = {}

        return obs, reward, done, info

    def _get_obs(self):
        min_dist = min(self.lidar_ranges) if self.lidar_ranges else 10.0
        curvature = self.curvatures[self.current_wp_idx] if self.current_wp_idx < self.numWaypoints else 0.0

        obs = np.array([min_dist, self.speed, curvature], dtype=np.float32)
        return obs

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()


def main():
    env = ControllerArbiterEnv()
    obs = env.reset()
    done = False
    while not done:
        action = 0  # Dummy action, replace with RL policy output
        obs, reward, done, info = env.step(action)

if __name__ == '__main__':
    main()

