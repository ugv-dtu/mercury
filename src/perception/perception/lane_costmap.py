import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
from std_msgs.msg import Header
from builtin_interfaces.msg import Time


class RoadLineCostmapNode(Node):

    def __init__(self):
        super().__init__('road_line_costmap_node')

        homography_matrix = [-0.9335399306051015,-1.3748497159820596,599.7909186871361,0.014278805626526487,-3.855277519162032,1200.2335645488997,1.2228824435258832e-05,-0.004479695571165812,1.0]

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('costmap_topic', '/perception/road_costmap')
        self.declare_parameter('costmap_frame', 'map')

        self.declare_parameter('homography', homography_matrix)

        self.declare_parameter('bev_width', 800)
        self.declare_parameter('bev_height', 600)

        self.declare_parameter('adaptive_block_size', 51)
        self.declare_parameter('adaptive_c', -25)
        self.declare_parameter('morph_kernel_size', 3)
        self.declare_parameter('min_brightness', 180)

        self.declare_parameter('resolution', 0.01)

        self.declare_parameter('origin_x', -4.0)
        self.declare_parameter('origin_y', -3.0)

        self.declare_parameter('undistort', True)
        self.declare_parameter('show_debug', True)

        self._read_params()

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            sensor_qos
        )

        self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            sensor_qos
        )

        costmap_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.costmap_pub = self.create_publisher(
            OccupancyGrid,
            self.costmap_topic,
            costmap_qos
        )

        self.get_logger().info("Road line costmap node started")

    def _read_params(self):

        self.image_topic = self.get_parameter('image_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.costmap_topic = self.get_parameter('costmap_topic').value
        self.costmap_frame = self.get_parameter('costmap_frame').value

        h_flat = self.get_parameter('homography').value
        self.homography = np.array(h_flat).reshape(3,3)

        self.bev_w = self.get_parameter('bev_width').value
        self.bev_h = self.get_parameter('bev_height').value

        self.block_size = self.get_parameter('adaptive_block_size').value
        self.adaptive_c = self.get_parameter('adaptive_c').value
        self.morph_k = self.get_parameter('morph_kernel_size').value
        self.min_bright = self.get_parameter('min_brightness').value

        self.resolution = self.get_parameter('resolution').value
        self.origin_x = self.get_parameter('origin_x').value
        self.origin_y = self.get_parameter('origin_y').value

        self.undistort = self.get_parameter('undistort').value
        self.show_debug = self.get_parameter('show_debug').value


    def camera_info_callback(self, msg):

        if self.camera_matrix is not None:
            return

        self.camera_matrix = np.array(msg.k).reshape(3,3)
        self.dist_coeffs = np.array(msg.d)

        self.get_logger().info("Camera calibration received")


    def image_callback(self, msg):

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(str(e))
            return

        if self.undistort and self.camera_matrix is not None:
            frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

        bev = cv2.warpPerspective(
            frame,
            self.homography,
            (self.bev_w, self.bev_h)
        )

        # centered crop
        bev = bev[:, :630]

        mask = self.detect_white_lines(bev)

        grid = self.mask_to_grid(mask, msg.header.stamp)

        self.costmap_pub.publish(grid)

        if self.show_debug:
            cv2.imshow("BEV", bev)
            cv2.imshow("Line Mask", mask)
            cv2.waitKey(1)


    def detect_white_lines(self, bev):

        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)

        _, bright = cv2.threshold(gray, self.min_bright, 255, cv2.THRESH_BINARY)

        bs = max(3, self.block_size | 1)

        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            bs,
            self.adaptive_c
        )

        mask = cv2.bitwise_and(bright, adaptive)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.morph_k, self.morph_k)
        )

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask


    def mask_to_grid(self, mask, stamp):

        grid = OccupancyGrid()
        grid.header = Header()
        grid.header.stamp = stamp
        grid.header.frame_id = self.costmap_frame

        # rotate to match ROS coordinate system
        rot = np.rot90(mask, k=-1)
        rot = np.flipud(rot)
        
        grid.info.resolution = self.resolution
        grid.info.width = rot.shape[1]
        grid.info.height = rot.shape[0]

        origin = Pose()
        origin.position.x = self.origin_x
        origin.position.y = self.origin_y
        origin.orientation.w = 1.0

        grid.info.origin = origin

        cost = np.where(rot > 0, 100, 0).astype(np.int8)

        grid.data = cost.flatten().tolist()

        return grid


def main():

    rclpy.init()

    node = RoadLineCostmapNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()