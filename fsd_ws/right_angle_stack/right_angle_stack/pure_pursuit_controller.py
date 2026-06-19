import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node

from .utils import normalize_angle, quaternion_to_yaw, world_to_body

# Pure Pursuit 路径跟踪控制器。


class PurePursuitController(Node):
    def __init__(self):
        super().__init__('pure_pursuit_controller')

        self.declare_parameter('target_speed', 3.0)
        self.declare_parameter('min_speed', 1.2)

        # 目标点必须在车前方且距离当前车体有一定距离。
        self.declare_parameter('lookahead_distance', 4.0)
        self.declare_parameter('min_lookahead_distance', 2.0)
        self.declare_parameter('max_yaw_rate', 1.4)
        self.declare_parameter('stop_distance', 1.2)
        self.declare_parameter('yaw_error_gain', 0.45)
        self.declare_parameter('path_heading_gain', 0.65)
        self.declare_parameter('turn_slowdown_factor', 0.35)
        self.declare_parameter('steer_speed_ratio', 0.75)

        self.target_speed = float(self.get_parameter('target_speed').value)
        self.min_speed = float(self.get_parameter('min_speed').value)
        self.lookahead_distance = float(self.get_parameter('lookahead_distance').value)
        self.min_lookahead = float(self.get_parameter('min_lookahead_distance').value)
        self.max_yaw_rate = float(self.get_parameter('max_yaw_rate').value)
        self.stop_distance = float(self.get_parameter('stop_distance').value)
        self.yaw_error_gain = float(self.get_parameter('yaw_error_gain').value)
        self.path_heading_gain = float(self.get_parameter('path_heading_gain').value)
        self.turn_slowdown_factor = float(self.get_parameter('turn_slowdown_factor').value)
        self.steer_speed_ratio = float(self.get_parameter('steer_speed_ratio').value)

        self.path = []
        self.state = None
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Path, '/planning/centerline', self.on_path, 10)
        self.create_subscription(Odometry, '/localization/odom', self.on_odom, 20)
        self.create_timer(0.05, self.on_timer)
        self.get_logger().info('Pure pursuit controller started.')

    def on_path(self, msg):
        self.path = [
            (pose.pose.position.x, pose.pose.position.y)
            for pose in msg.poses
        ]

    def on_odom(self, msg):
        self.state = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            quaternion_to_yaw(msg.pose.pose.orientation),
        )

    def publish_stop(self):
        self.cmd_pub.publish(Twist())

    def nearest_forward_index(self, x, y, yaw):
        best_index = None
        best_dist = float('inf')
        for i, (px, py) in enumerate(self.path):
            dx = px - x
            dy = py - y
            local_x, _ = world_to_body(dx, dy, yaw)
            if local_x < 0.0:
                continue
            dist = math.hypot(dx, dy)
            if dist < best_dist:
                best_dist = dist
                best_index = i
        if best_index is not None:
            return best_index
        return min(
            range(len(self.path)),
            key=lambda i: math.hypot(self.path[i][0] - x, self.path[i][1] - y),
        )

    def path_heading(self, index):
        if index + 1 >= len(self.path):
            index = max(0, index - 1)
        if index + 1 >= len(self.path):
            return 0.0
        dx = self.path[index + 1][0] - self.path[index][0]
        dy = self.path[index + 1][1] - self.path[index][1]
        if abs(dx) + abs(dy) < 1e-6:
            return 0.0
        return math.atan2(dy, dx)

    def upcoming_bend(self, start_index):
        if start_index >= len(self.path) - 2:
            return 0.0
        base = self.path_heading(start_index)
        max_delta = 0.0
        limit = min(len(self.path) - 2, start_index + 18)
        for i in range(start_index + 1, limit + 1):
            max_delta = max(max_delta, abs(normalize_angle(self.path_heading(i) - base)))
        return max_delta

    def on_timer(self):
        if self.state is None or len(self.path) < 2:
            self.publish_stop()
            return

        x, y, yaw = self.state

        nearest_index = self.nearest_forward_index(x, y, yaw)

        goal_x, goal_y = self.path[-1]

        if nearest_index >= len(self.path) - 4 and math.hypot(goal_x - x, goal_y - y) < self.stop_distance:
            self.publish_stop()
            return

        bend = self.upcoming_bend(nearest_index)
        bend_scale = min(1.0, bend / 0.25)
        effective_lookahead = self.lookahead_distance - bend_scale * (
            self.lookahead_distance - self.min_lookahead
        )

        target = self.path[-1]
        target_index = len(self.path) - 2
        for index, point in enumerate(self.path[nearest_index:], start=nearest_index):
            dx = point[0] - x
            dy = point[1] - y
            local_x, _ = world_to_body(dx, dy, yaw)
            if local_x > 0.2 and math.hypot(dx, dy) >= effective_lookahead:
                target = point
                target_index = index
                break

        local_x, local_y = world_to_body(target[0] - x, target[1] - y, yaw)
        lookahead = max(0.5, math.hypot(local_x, local_y))

        curvature = 2.0 * local_y / (lookahead * lookahead)
        yaw_error = math.atan2(local_y, max(local_x, 1e-3))
        heading_index = min(
            len(self.path) - 2,
            target_index + max(1, int(2 + 3 * bend_scale)),
        )
        path_yaw = self.path_heading(heading_index)
        heading_error = normalize_angle(path_yaw - yaw)

        turn_slowdown = min(1.0, abs(curvature) / 0.28)
        speed = self.target_speed * (1.0 - self.turn_slowdown_factor * turn_slowdown)
        speed = max(self.min_speed, speed)

        steer_speed = max(speed, self.target_speed * self.steer_speed_ratio)
        yaw_rate = steer_speed * curvature
        yaw_rate += self.yaw_error_gain * normalize_angle(yaw_error)
        yaw_rate += self.path_heading_gain * heading_error * (1.0 + 0.75 * bend_scale)
        yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, yaw_rate))

        cmd = Twist()
        cmd.linear.x = speed
        cmd.angular.z = yaw_rate
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.publish_stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
