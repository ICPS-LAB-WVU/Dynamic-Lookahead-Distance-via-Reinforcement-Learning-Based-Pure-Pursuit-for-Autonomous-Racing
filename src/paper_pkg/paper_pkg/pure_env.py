# Filename: paper_pkg/pure_env.py

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from sensor_msgs.msg import LaserScan
import gym
from gym import spaces
import numpy as np
import time
import random
import os
from tf_transformations import quaternion_from_euler

class F1TenthEnv(gym.Env):
    def __init__(self):
        if not rclpy.ok():  # ✅ Prevent double initialization
            rclpy.init()

        super().__init__()
        self.node = rclpy.create_node('rl_lookahead_env')

        # === Action Space ===

        # Action: lookahead distance (0.5 to 2.0 meters)
        self.action_space = spaces.Box(low=np.array([0.35]), high=np.array([4.0]), dtype=np.float32)

        # future-curvature taps (tune to your waypoint spacing; 5~12 ≈ 1–4 m ahead)
        self.k_offsets = [0, 5, 12]

        # observation now: [speed, k0, k1, k2, dk]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, -1.0]),
            high=np.array([12.0, 0.5, 0.5, 0.5, 1.0]),
            dtype=np.float32
        )

        # === Load Waypoints and Curvature ===

        map_path = os.path.abspath(os.path.join('src/paper_ws/src', 'csv_data'))
        csv_data = np.loadtxt(f'{map_path}/Austin_fast.csv', delimiter=';', skiprows=3)
        self.waypoints = csv_data[:, 1:3]
        self.ref_speed = csv_data[:, 5] * 1.5
        self.curvatures = csv_data[:, 4]  # kappa_radpm
        self.numWaypoints = self.waypoints.shape[0]

        # === Initial States ===

        self.speed = 0.0
        self.curvature = 0.0
        self.lookahead_distance = 1.0
        self.prev_lookahead = 1.0
        self.step_count = 0
        self.max_steps = 10000
        self.stalled_steps = 0
        self.stall_limit = 200
        self.collision = False
        self.current_wp_idx = 0
        self.last_wp_idx = 0
        self.lap_count = 0

        # === ROS Topics ===

        self.lookahead_pub = self.node.create_publisher(Float32, '/lookahead_distance', 10)
        self.odom_sub = self.node.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.reset_pub = self.node.create_publisher(Float32, '/episode_reset_signal', 10)
        self.scan_sub = self.node.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.pose_sub = self.node.create_subscription(Odometry, '/ego_racecar/odom', self.pose_callback, 10)
        

    def odom_callback(self, msg):
        self.speed = msg.twist.twist.linear.x

    def scan_callback(self, msg):
        if len(msg.ranges) > 0 and min(msg.ranges) < 0.2:
            self.collision = True

    def curvature_features(self):
        idxs = [(self.current_wp_idx + o) % self.numWaypoints for o in self.k_offsets]
        ks = np.abs(self.curvatures[idxs])
        k0, k1, k2 = ks
        dk = k1 - k0
        kmax = float(np.max(ks))
        return k0, k1, k2, dk, kmax
    
    def _obs(self):
        k0, k1, k2, dk, kmax = self.curvature_features()
        return np.array([self.speed, k0, k1, k2, dk], dtype=np.float32)

    def pose_callback(self, msg):
        if self.waypoints is None or len(self.waypoints) == 0:
            return
        car_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        dists = np.linalg.norm(self.waypoints - car_pos, axis=1)
        closest_idx = np.argmin(dists)
        if closest_idx > self.current_wp_idx:
            self.current_wp_idx = closest_idx

        # ✅ Update curvature from CSV
        # ✅ Smoothed curvature update here
        alpha = 0.8
        raw_curvature = self.curvatures[self.current_wp_idx]
        self.curvature = alpha * self.curvature + (1 - alpha) * raw_curvature

        #self.node.get_logger().info(
         #   f"[Pose Callback] WP idx: {self.current_wp_idx}, raw curvature: {raw_curvature:.6f}, smoothed: {self.curvature:.6f}"
        #)

    def reset(self):
        self.lookahead_distance = 1.0
        self.prev_lookahead = 1.0
        self.step_count = 0
        self.collision = False
        self.stalled_steps = 0
        self.current_wp_idx = 0
        self.last_wp_idx = 0


        self.reset_car_position()

        # NEW: Publish reset signal
        reset_msg = Float32()
        reset_msg.data = 1.0  # Arbitrary signal
        self.reset_pub.publish(reset_msg)

        obs = self._obs()
        return obs

    def reset_car_position(self):
        reset_pub = self.node.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'

        # CHANGE: Reset from full lap range
        # 70% of the time, start near a sharp bend to learn corners fast
        if random.random() < 0.70:
            candidates = np.where(np.abs(self.curvatures) > 0.06)[0]
            idx = int(random.choice(candidates)) if len(candidates) > 0 else random.randint(0, len(self.waypoints) - 2)
        else:
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

    def step(self, action):
        action = np.clip(action, 0.35, 4.0)
        lookahead = float(action[0])

        # (optional) light smoothing to reduce jitter
        beta = 0.2
        lookahead = beta * lookahead + (1 - beta) * self.prev_lookahead
        self.lookahead_distance = lookahead

        msg = Float32()
        msg.data = lookahead
        self.lookahead_pub.publish(msg)

        rclpy.spin_once(self.node, timeout_sec=0.1)

         # ----- curvature lookahead features -----
        k0, k1, k2, dk, kmax = self.curvature_features()

        # Ideal L shrinks with upcoming curvature, grows with speed
        ideal_lookahead = 0.50 + 0.28 * self.speed - 3.5 * kmax
        ideal_lookahead = float(np.clip(ideal_lookahead, 0.35, 4.0))

        lookahead_error = abs(self.lookahead_distance - ideal_lookahead)

        # Log the chosen action
        if self.step_count % 1000 == 0:
            self.node.get_logger().info(f"[Step {self.step_count}] Speed={self.speed:.2f} m/s, Curv={self.curvature:.5f}, "
                f"IdealL={ideal_lookahead:.2f}, ActionL={lookahead:.2f}")
            
        # --- Reward Design ---
        reward = 0.0

        # ✅ Reward speed (faster is better)
        reward += 1.8 * self.speed

        # ✅ Penalize deviation from ideal lookahead (steering mismatch)
        reward -= 3.0 * lookahead_error # track the ideal

        # ✅ Penalize sudden steering changes (jerkiness)
        reward -= 0.4 * abs(self.lookahead_distance - self.prev_lookahead)  # smoothness

        # ✅ Penalize high curvature (sharp turns)
        reward -= 1.5 * abs(self.curvature)

        reward -= 2.0 * (lookahead * kmax)                     # long-L near high curvature is bad
        if kmax > 0.08 and lookahead <= (1.2 + 0.05 * self.speed):
            reward += 1.5  # early braking effect: pre-shorten L before the bend

        if self.speed >= 5.5 and abs(self.curvature) < 0.02:
            L_min_straight = 0.5 + 0.35 * self.speed  # e.g., ~3.4 m at 8 m/s
            L_min_straight = float(np.clip(L_min_straight, 1.5, 4.0))
            if self.lookahead_distance >= L_min_straight:
                reward += 1.0
            else:
                reward -= 1.0
        
        # ✅ Penalize collision
        if self.collision:
            reward -= 10.0

        # ✅ Penalize crawling
        if self.speed < 0.1:
            reward -= 0.5  # penalize crawling

        # ✅ Reward progress (waypoints advanced)
        if self.current_wp_idx > self.last_wp_idx:
            reward += 1.0 * (self.current_wp_idx - self.last_wp_idx)
            self.last_wp_idx = self.current_wp_idx

        # ✅ Penalize staying near start
        #if self.current_wp_idx < 0.1 * self.numWaypoints:
         #   reward -= 1.0

        # === Done conditions ===
        if self.speed < 0.05:
            self.stalled_steps += 1
        else:
            self.stalled_steps = 0

        self.prev_lookahead = self.lookahead_distance
        self.step_count += 1

        done = (
            self.collision or
            self.stalled_steps >= self.stall_limit or
            self.step_count >= self.max_steps
        )

        # ✅ Lap completed check (looped to start after almost full lap)
        if self.current_wp_idx < self.last_wp_idx:
            self.node.get_logger().info("Episode ended: Lap completed")
            reward += 20.0
            done = True

        if done:
            if self.collision:
                self.node.get_logger().info("Episode ended: Collision")
            elif self.stalled_steps >= self.stall_limit:
                self.node.get_logger().info("Episode ended: Vehicle stalled")
            elif self.step_count >= self.max_steps:
                self.node.get_logger().info("Episode ended: Max steps reached")
                reward += 20.0  # ✅ Bonus for surviving till the end

        reward = np.clip(reward, -20.0, 50.0)

        obs = self._obs()
        return obs, reward, done, {}    

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()