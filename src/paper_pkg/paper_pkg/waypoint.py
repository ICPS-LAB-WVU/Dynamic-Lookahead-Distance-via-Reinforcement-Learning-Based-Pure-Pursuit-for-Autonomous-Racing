#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
import numpy as np
import atexit
from time import gmtime, strftime
from numpy import linalg as LA
from tf_transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import pandas as pd
from scipy.interpolate import splprep, splev
import matplotlib
matplotlib.use("Agg")  # headless backend for Docker
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from pathlib import Path

class WaypointsLogger(Node):
    def __init__(self):
        super().__init__('waypoints_logger')

        # --- Parameters ---
        self.declare_parameter('is_real', False)
        self.declare_parameter('min_spacing', 0.01)
        # where to save files; override with ROS param or env WAYPOINTS_DIR
        default_out = os.environ.get('WAYPOINTS_DIR', '/sim_ws/output')
        self.declare_parameter('output_dir', default_out)

        self.is_real = self.get_parameter('is_real').value
        self.min_spacing = float(self.get_parameter('min_spacing').value)
        self.output_dir = self.get_parameter('output_dir').value

        # ensure output dir exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        ts = strftime('%Y-%m-%d-%H-%M', gmtime())
        self.filename = os.path.join(self.output_dir, f'wp-{ts}.csv')
        self.fig_path = self.filename.replace('.csv', '.png')

        # open file now (not at import time)
        self.file = open(self.filename, 'w', buffering=1)
        self.file.write('# x_m, y_m, w_tr_right_m, w_tr_left_m\n')
        self.get_logger().info(f'Logging to {self.filename}')

        # --- Topics ---
        odom_topic = '/pf/pose/odom' if self.is_real else '/ego_racecar/odom'
        self.subscription_odom = self.create_subscription(
            Odometry, odom_topic, self.process_odometry, 10)
        self.subscription_scan = self.create_subscription(
            LaserScan, '/scan', self.process_scan, 10)

        # --- State ---
        self.latest_scan = False
        self.latest_odometry = None
        self.previous_point = None
        self.waypoints = []  # (x, y, left_width, right_width)
        self.left_width = np.nan
        self.right_width = np.nan

    def process_scan(self, scan_data: LaserScan):
        """Estimate track widths from LiDAR data."""
        ranges = np.array(scan_data.ranges, dtype=float)
        angles = np.linspace(scan_data.angle_min, scan_data.angle_max, len(ranges))

        valid = np.isfinite(ranges) & ~np.isnan(ranges)
        if not np.any(valid):
            self.get_logger().warn('No valid ranges in scan data.')
            return

        ranges = ranges[valid]
        angles = angles[valid]

        left_mask = angles > 0
        right_mask = ~left_mask  # <= 0

        # Percentile is more stable than min
        self.left_width = (np.percentile(ranges[left_mask], 20)
                           if np.any(left_mask) else float('inf'))
        self.right_width = (np.percentile(ranges[right_mask], 20)
                            if np.any(right_mask) else float('inf'))

        self.latest_scan = True
        self.save_waypoint()

    def process_odometry(self, odometry_data: Odometry):
        self.latest_odometry = odometry_data
        self.save_waypoint()

    def save_waypoint(self):
        """Save a waypoint if both LiDAR and odometry data are available."""
        if self.latest_scan and (self.latest_odometry is not None):
            data = self.latest_odometry
            x = data.pose.pose.position.x
            y = data.pose.pose.position.y
            quat = [
                data.pose.pose.orientation.x,
                data.pose.pose.orientation.y,
                data.pose.pose.orientation.z,
                data.pose.pose.orientation.w
            ]
            yaw = euler_from_quaternion(quat)[2]  # not used now, but keep for future

            if (self.previous_point is None or
                LA.norm([x - self.previous_point[0], y - self.previous_point[1]]) >= self.min_spacing):
                self.get_logger().info(
                    f'Saving waypoint: x={x:.2f}, y={y:.2f}, '
                    f'left_width={self.left_width:.2f}, right_width={self.right_width:.2f}'
                )
                self.waypoints.append((x, y, float(self.left_width), float(self.right_width)))
                self.previous_point = (x, y)

            # reset flags
            self.latest_scan = False
            self.latest_odometry = None

    def filter_outliers(self, points, threshold=2.0):
        """Filter waypoints far from the moving last-accepted point."""
        if len(points) < 3:
            return np.asarray(points)
        filtered = [points[0]]
        for i in range(1, len(points)):
            if LA.norm(np.asarray(points[i]) - np.asarray(filtered[-1])) < threshold:
                filtered.append(points[i])
        return np.asarray(filtered)

    def save_and_interpolate(self):
        """Filter, interpolate, save to CSV, and plot the trajectory."""
        if len(self.waypoints) > 1:
            wp = np.asarray(self.waypoints, dtype=float)
            x, y = wp[:, 0], wp[:, 1]
            left_w, right_w = wp[:, 2], wp[:, 3]

            # Filter outliers (on geometry only)
            filtered_xy = self.filter_outliers(wp[:, :2], threshold=2.0)
            x_filt, y_filt = filtered_xy[:, 0], filtered_xy[:, 1]

            # Guard for duplicate points that can break splprep
            if len(filtered_xy) >= 3 and np.unique(filtered_xy, axis=0).shape[0] >= 3:
                tck, _ = splprep([x_filt, y_filt], s=0.2)
                unew = np.linspace(0, 1, 5000)
                x_new, y_new = splev(unew, tck)
            else:
                # fallback: just use filtered points
                x_new, y_new = x_filt, y_filt
                self.get_logger().warn('Too few unique points for spline; using filtered points.')

            avg_left = float(np.nanmean(left_w))
            avg_right = float(np.nanmean(right_w))

            for xi, yi in zip(x_new, y_new):
                self.file.write(f'{xi}, {yi}, {avg_right}, {avg_left}\n')

            self.get_logger().info(f'Waypoints saved: {len(x_new)} points -> {self.filename}')

            # Save plot (no blocking show)
            plt.figure()
            plt.plot(x, y, 'o', label='Original Waypoints')
            plt.plot(x_filt, y_filt, 'x', label='Filtered Waypoints')
            plt.plot(x_new, y_new, '-', label='Interpolated Path')
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.title('Waypoint Path')
            plt.legend()
            plt.grid()
            plt.savefig(self.fig_path, dpi=150)
            plt.close()

            if len(x_new) < 10:
                self.get_logger().warn('Generated trajectory has fewer points than the required horizon (10).')

        # always close the file here
        try:
            self.file.close()
        except Exception:
            pass

def main(args=None):
    print('Starting waypoint logger...')
    rclpy.init(args=args)
    node = WaypointsLogger()

    def _shutdown():
        print('Goodbye. Files saved.')
    atexit.register(_shutdown)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_and_interpolate()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

