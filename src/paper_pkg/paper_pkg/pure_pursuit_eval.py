#!/usr/bin/env python3
import os
import collections
import numpy as np
import rclpy
from scipy.spatial import distance, transform
from rclpy.node import Node

from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped

import gym
from gym import spaces
import torch
from stable_baselines3 import PPO


class PurePursuit(Node):
    """Pure Pursuit with RL-selected lookahead; speed is taken from CSV (racing)."""

    # -------------- init --------------
    def __init__(self):
        super().__init__('pure_pursuit_rl_eval')

        # === run mode ===
        self.is_real = False
        self.map_name = 'YasMarina_fast'
        self.is_ascending = True
        self.prev_steering = 0.0  # for smoothing

        # Topics
        drive_topic = '/drive'
        odom_topic = '/pf/viz/inferred_pose' if self.is_real else '/ego_racecar/odom'
        visualization_topic = '/visualization_marker_array'

        # Publishers & Subscribers
        self.sub_pose = self.create_subscription(PoseStamped if self.is_real else Odometry, odom_topic, self.pose_callback, 1)
        self.pub_drive = self.create_publisher(AckermannDriveStamped, drive_topic, 1)
        self.pub_vis = self.create_publisher(MarkerArray, visualization_topic, 1)

        self.drive_msg = AckermannDriveStamped()
        self.markerArray = MarkerArray()

        self.k_offsets = [0, 5, 12]  # indices ahead used in training


        # === load CSV ===
        map_path = os.path.abspath(os.path.join('src/paper_ws/src', 'csv_data'))
        csv = np.loadtxt(os.path.join(map_path, f'{self.map_name}.csv'), delimiter=';', skiprows=3)
        self.waypoints   = csv[:, 1:3]   # x,y
        self.curvatures  = csv[:, 4]     # kappa
        self.ref_speed   = csv[:, 5]  * 1.15   # m/s (commanded speed)
        self.numWaypoints = self.waypoints.shape[0]

        # precompute path headings (for robust seeding)
        p0 = self.waypoints
        p1 = np.roll(self.waypoints, -1, axis=0)
        d  = p1 - p0
        self.path_yaw = np.arctan2(d[:, 1], d[:, 0])

        # === state ===
        self.currX = 0.0
        self.currY = 0.0
        self.rot   = np.eye(3)           # body->world rotation
        self.speed = 0.0
        self.kappa_ema = 0.0
        self.current_wp_idx = 0
        self.min_front_range = 10.0

        # Lookahead & stability guards
        self.prev_L = 1.5
        self.L = 1.5
        self.straight_kappa_thresh = 0.02  # ≈ 

        # histories
        self._idx_hist = collections.deque(maxlen=20)
        self._pos_hist = collections.deque(maxlen=20)

        # === RL model (lookahead only) ===
        model_path = '/sim_ws/src/paper_ws/src/test2/ppo_lookahead_model4.zip'
        if not os.path.exists(model_path):
            self.get_logger().warn(f"Model not found at {model_path}, trying /mnt/data...")
            alt = '/mnt/data/ppo_lookahead_model.zip'
            model_path = alt if os.path.exists(alt) else model_path
        self.model = PPO.load(model_path, env=self._make_dummy_env())

        # VecNormalize stats (optional)
        self.obs_mean = None
        self.obs_var  = None
        self._maybe_load_vecnorm_stats([
            './vecnorm.pkl',
            '/mnt/data/vecnorm.pkl',
            '/sim_ws/src/paper_ws/src/test2/vecnorm4.pkl'
        ])

        self.visualization_init()
        self.get_logger().info("✅ Eval: speed from CSV, lookahead from RL, stabilized like trainer.")

    # -------------- helper env for loading --------------
    def _make_dummy_env(self):
        class EvalEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.action_space = spaces.Box(low=np.array([0.35]),
                                               high=np.array([4.0]),
                                               dtype=np.float32)
                self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, -1.0]),
                                                    high=np.array([12.0,  0.5, 0.5, 0.5, 1.0]),
                                                    dtype=np.float32)
            def reset(self, **kwargs):
                return np.zeros(self.observation_space.shape, dtype=np.float32)
            def step(self, action):
                return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, {}
        return EvalEnv()

    def _maybe_load_vecnorm_stats(self, try_paths):
        from stable_baselines3.common.vec_env import VecNormalize
        for p in try_paths:
            try:
                vec = VecNormalize.load(p)
                self.obs_mean = vec.obs_rms.mean.copy()
                self.obs_var  = vec.obs_rms.var.copy()
                self.get_logger().info(f"Loaded VecNormalize stats: {p}")
                return
            except Exception:
                continue
        self.get_logger().warn("VecNormalize stats not found; continuing without obs normalization (actions may be off).")

    def _normalize_obs(self, obs):
        if getattr(self, "obs_mean", None) is None or getattr(self, "obs_var", None) is None:
            return obs
        return (obs - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)

    def build_control_gain(self, speed, v_min=3.427, v_max=13.8, gain_min=0.55, gain_max=0.6):
        m = (gain_min - gain_max) / (v_max - v_min)
        b = gain_max - m * v_min
        return float(np.clip(m * speed + b, gain_min, gain_max))

    # --------- helpers ----------
    def max_abs_curvature_ahead(self, arc_len=2.0, start_idx=None):
        """Max |kappa| over arc_len (m) ahead along waypoints starting near closest index."""
        if start_idx is None:
            start_idx = getattr(self, "closest_index", 0)

        steps = 1
        avg_ds = float(np.mean(np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)))
        if avg_ds > 1e-6:
            steps = max(1, int(arc_len / avg_ds))
        idxs = (start_idx + np.arange(steps)) % self.numWaypoints
        return float(np.max(np.abs(self.curvatures[idxs])))

    def get_lookahead_point(self, threshold):
        idx = self.closest_index
        # move forward until distance >= threshold
        while self.distances[idx] < threshold:
            idx = (idx + 1) % self.numWaypoints if self.is_ascending else (idx - 1 + self.numWaypoints) % self.numWaypoints
        return self.waypoints[idx]

    def transform_to_vehicle_frame(self, point):
        # local = R^T * (point - position)
        pvect = (point - np.array([self.currX, self.currY]))
        local3 = self.rot.T @ np.array([pvect[0], pvect[1], 0.0])
        return np.array([local3[0], local3[1], 0.0, 0.0])
    

    def curvature_features_eval(self):
        idx0 = self.closest_index
        n = self.numWaypoints
        k = self.curvatures
        idxs = [(idx0 + o) % n for o in self.k_offsets]
        k0, k1, k2 = [abs(float(k[i])) for i in idxs]
        dk = k1 - k0
        kmax = max(k0, k1, k2)
        return k0, k1, k2, dk, kmax


    # -------------- Odom --------------
    def pose_callback(self, msg):
        # Get position & orientation
        if self.is_real:
            pos = msg.pose
            quat = pos.orientation
        else:
            pos = msg.pose.pose
            quat = pos.orientation
            # Read speed from Odometry when available
            try:
                self.speed = float(msg.twist.twist.linear.x)
            except Exception:
                pass

        self.currX = pos.position.x
        self.currY = pos.position.y

        Rmat = transform.Rotation.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()
        self.rot = Rmat

        # Find closest waypoint
        currPos = np.array([[self.currX, self.currY]])
        self.distances = distance.cdist(currPos, self.waypoints).reshape((self.numWaypoints))
        self.closest_index = int(np.argmin(self.distances))
        self.closestPoint = self.waypoints[self.closest_index]
        speed_cmd = float(self.ref_speed[self.closest_index])

        # Update curvature & EMA (smoother obs for RL)
        k_now = float(self.curvatures[self.closest_index])

        # Update steering gain
        self.steering_gain = 0.6

        # ---- RL: observation → lookahead ----
        k0, k1, k2, dk, kmax = self.curvature_features_eval()
        obs = np.array([self.speed, k0, k1, k2, dk], dtype=np.float32)
        obs_infer = self._normalize_obs(obs)
        with torch.no_grad():
            action, _ = self.model.predict(obs_infer, deterministic=True)

        L_pred = float(np.asarray(action).reshape(-1)[0])
        L_rl = float(np.clip(L_pred, 0.35, 4.0))  # match training bounds

        # Curvature-aware ceiling (engages mainly in hairpins)
        kappa_ahead = self.max_abs_curvature_ahead(arc_len=max(1.0, 1.5 * max(L_rl, 1.0)))
        L_cap = float(np.clip(1.3 / np.sqrt(kappa_ahead + 1e-6), 0.35, 2.5))
        L = min(L_rl, L_cap)

        # Straight-line minimum (stability guardrail)
        if self.speed >= 6.3 and kappa_ahead < self.straight_kappa_thresh:
            L_straight_min = float(np.clip(0.35 + 0.35 * self.speed, 1.5, 2.5))
            L = max(L, L_straight_min)

        # L smoothing (stronger on straights to avoid jitter)
        if kappa_ahead < self.straight_kappa_thresh:
            L = 0.6 * L + 0.4 * self.prev_L
        else:
            L = 0.85 * L + 0.15 * self.prev_L
        self.prev_L = L
        self.L = L

        # Get target point and transform
        target = self.get_lookahead_point(self.L)
        translated = self.transform_to_vehicle_frame(target)

        # Compute curvature and steering angle (bicycle PP)
        y = translated[1]
        gamma = self.steering_gain * (2.0 * y / max(self.L, 1e-3)**2)
        gamma = float(np.clip(gamma, -0.35, 0.35))

        # Smooth steering using a low-pass filter
        alpha = 0.4
        gamma = (1 - alpha) * self.prev_steering + alpha * gamma
        self.prev_steering = gamma

        # publish
        self.drive_msg.drive.steering_angle = gamma
        self.drive_msg.drive.speed = speed_cmd if not self.is_real else min(speed_cmd, 2.0)
        self.pub_drive.publish(self.drive_msg)

        # viz
        self.targetMarker.points = [Point(x=target[0], y=target[1], z=0.0)]
        self.closestMarker.points = [Point(x=self.closestPoint[0], y=self.closestPoint[1], z=0.0)]
        self.markerArray.markers = [self.waypointMarker, self.targetMarker, self.closestMarker]
        self.pub_vis.publish(self.markerArray)

        # logs
        self.get_logger().info(
            f"idx={self.closest_index:04d}, pos=({self.currX:.2f},{self.currY:.2f}), "
            f"L_rl={L_rl:.2f}, L_cap={L_cap:.2f}, L={self.L:.2f}, "
            f"steer={gamma:.2f}, v_csv={speed_cmd:.2f}, v_odom={self.speed:.2f}, "
            f"kappa_ahead={kappa_ahead:.5f}"
        )

    # -------------- viz --------------
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


# -------------- main --------------
def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
