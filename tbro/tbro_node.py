"""
Transformer Based Radar Odometry ROS node
"""

# generic
import rclpy
from rclpy.node import Node
import math
import numpy as np
from tf_transformations import quaternion_from_euler
import scipy.spatial.transform as tf

# replace this with dca1000_device/msg/MimoMsg
# ros messages
from std_msgs.msg import String
from dca1000_device.msg import MimoMsg
from nav_msgs.msg import Odometry

# pytorch
import torch
from torch.utils.data import DataLoader


# Custom dataset for TBRO
from .mimo_dataset import MimoDataset

# local
# from .models.deep_ro_enc_only import DeepROEncOnly
from .parameters import Parameters
from .models.kremer_original import KramerOriginal


class TbroSubscriber(Node):

    def __init__(self):
        super().__init__("tbro_subscriber")
        self.controller_period = 0.1
        self.init_flag = False  # False if not initialzied
        self.msg_buffer = []
        self.data_set = MimoDataset()
        self.pose = np.zeros(6)
        self.args = Parameters()
        self.device = torch.device("cpu")
        # self.model = DeepROEncOnly(self.args)
        self.model = KramerOriginal(
            self.args,
            "/home/parallels/radar/models/epoch_25_batch_16_lr_1e-05_tbro_test_batch.model",
            # "/home/parallels/radar/models/motion_only_epoch_25_batch_10_lr_1e-05_tbro_test_batch.model",
            # "/home/parallels/radar/models/no_motion_epoch_10_batch_4_lr_1e-05_test_batch.model",
            # "/home/parallels/radar/models/second_motion_only_epoch_50_batch_10_lr_1e-05_tbro_test_batch.model",
        )
        # self.model.run
        self.model.to(self.device)

        # heatmap subscriber
        self.heatmap_subcriber = self.create_subscription(
            MimoMsg, "/cascade/heatmap", self.listener_callback, 10
        )

        self.heatmap_subcriber  # prevent unused variable warning
        # Timer callback
        self.timer = self.create_timer(self.controller_period, self.timer_callback)
        self.odometry_publisher = self.create_publisher(Odometry, "tbro_odometry", 10)

        # TODO: Check if I can append first two messages here instad of in listenter_callback

    def listener_callback(self, msg):
        # self.get_logger().info('I got heatmap image: "%s"' % msg)
        # inti the system when first ever message is received
        if not self.init_flag:
            self.data_set.load_img(msg)
            if self.data_set.__len__() == 2:
                self.init_flag = True
        else:
            self.msg_buffer.append(msg)

    # Timer callback function
    def timer_callback(self):
        # print("Timer")

        self.get_logger().debug("Running tbro...")
        self.run_once()

    def to_transform_torch(
        self, positions: torch.Tensor, orientations: torch.Tensor
    ) -> torch.Tensor:
        batch_size = positions.shape[0]
        seq_len = positions.shape[1]

        poses_mat = torch.zeros(
            batch_size, seq_len, 4, 4, dtype=positions.dtype, device=positions.device
        )

        poses_mat[:, :, 3, 3] += 1
        poses_mat[:, :, :3, 3] += positions
        for i, j in enumerate(orientations):
            for k, h in enumerate(j):
                rot_mat = tf.Rotation.from_euler("xyz", h.tolist())
                poses_mat[i, k, :3, :3] += torch.tensor(
                    rot_mat.as_matrix(), device=positions.device
                )
        return poses_mat

    # modifiy this function to transform odom by using two matrix
    # body frame word frame, transform trees
    def odometry_to_track(self, poses_mat: torch.Tensor) -> torch.Tensor:
        # Shape (batch_size, sequence_length, 6dof (xyz,rpq))
        if len(poses_mat.shape) == 4:
            batch_size = poses_mat.shape[0]
            seq_len = poses_mat.shape[1]

            first_pose = torch.tile(
                torch.eye(4, dtype=poses_mat.dtype, device=poses_mat.device).unsqueeze(
                    0
                ),
                (batch_size, 1, 1),
            )

            track = [first_pose]
            start_index = 1

            for i in range(start_index, seq_len):
                pose = poses_mat[:, i]
                prev = track[-1]

                track.append(torch.matmul(prev, pose))

            track = torch.stack(track, dim=1)
        if len(poses_mat.shape) == 3:
            print("Track without batching")
            seq_len = poses_mat.shape[0]

            first_pose = torch.tile(
                torch.eye(4, dtype=poses_mat.dtype, device=poses_mat.device).unsqueeze(
                    0
                ),
                (1, 1),
            )

            track = [first_pose]
            start_index = 1

            for i in range(start_index, seq_len):
                pose = poses_mat[i]
                prev = track[-1]

                track.append(torch.matmul(prev, pose))

            track = torch.stack(track, dim=1)
            print(track.shape)
        return track

    def homogeneous_transform(self, translation, rotation):
        """
        Generate a homogeneous transformation matrix from a translation vector
        and a quaternion rotation.

        Parameters:
        - translation: 1D NumPy array or list of length 3 representing translation along x, y, and z axes.
        - rotation: 1D NumPy array or list of length 4 representing a quaternion rotation.

        Returns:
        - 4x4 homogeneous transformation matrix.
        """

        # Ensure that the input vectors have the correct dimensions
        translation = np.array(translation, dtype=float)
        rotation = np.array(rotation, dtype=float)

        if translation.shape != (3,) or rotation.shape != (4,):
            raise ValueError(
                "Translation vector must be of length 3, and rotation quaternion must be of length 4."
            )

        # Normalize the quaternion to ensure it is a unit quaternion
        rotation /= np.linalg.norm(rotation)

        # Create a rotation matrix from the quaternion using scipy's Rotation class
        rotation_matrix = tf.Rotation.from_quat(rotation).as_matrix()

        # Create a 4x4 homogeneous transformation matrix
        homogeneous_matrix = np.eye(4)
        homogeneous_matrix[:3, :3] = rotation_matrix
        homogeneous_matrix[:3, 3] = translation

        return homogeneous_matrix

    def process_data(self):
        self.get_logger().debug("Processing frames")

        # Get displasment and rotation form ML model
        with torch.no_grad():
            pose_delta = self.model.forward(
                [self.data_set.__getitem__(0)[0], self.data_set.__getitem__(1)[0]]
            )

        # print(f'[{float(pose_delta[0][0])}, {float(pose_delta[0][1])}, {float(pose_delta[0][2])}, {float(pose_delta[0][3])}, {float(pose_delta[0][4])}, {float(pose_delta[0][5])}]')

        # Add delta pose to current pose
        self.pose[0] += float(pose_delta[0][0])
        self.pose[1] += float(pose_delta[0][1])
        self.pose[2] += float(pose_delta[0][2])
        self.pose[3] += float(pose_delta[0][3])
        self.pose[4] += float(pose_delta[0][4])
        self.pose[5] += float(pose_delta[0][5])

        # convert pose to 4x1 metrix
        position_body = self.pose[:3]
        print("positon_body: ", position_body)
        pose_body_frame = np.asmatrix(np.append(position_body, [1.0])).transpose()

        # Generate 4x4 homogeneous transformation metrix
        trasformation = self.pose[:3]
        rotation = quaternion_from_euler(
            float(self.pose[3]), float(self.pose[4]), float(self.pose[5])
        )
        rotation = np.asarray(rotation)
        ht_matrix = self.homogeneous_transform(trasformation, rotation)

        # apply transfoirmation
        pose_world = ht_matrix * pose_body_frame

        # Build odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "world"
        odom_msg.child_frame_id = "imu_link_enu"
        odom_msg.pose.pose.position.x = float(pose_world[0][0])
        odom_msg.pose.pose.position.y = float(pose_world[1][0])
        odom_msg.pose.pose.position.z = float(pose_world[2][0])
        odom_msg.pose.pose.orientation.x = rotation[0]
        odom_msg.pose.pose.orientation.y = rotation[1]
        odom_msg.pose.pose.orientation.z = rotation[2]
        odom_msg.pose.pose.orientation.w = rotation[3]

        print("Publishing pose: ", odom_msg.pose.pose)
        self.odometry_publisher.publish(odom_msg)

    def run_once(self):
        if self.init_flag:
            # TODO: This will miss the first message. Fix it.
            if len(self.msg_buffer) > 0:
                self.data_set.load_img(self.msg_buffer.pop(0))
                self.process_data()


def main(args=None):
    # print('Hi from tbro.')
    rclpy.init(args=args)

    tbro_subscriber = TbroSubscriber()

    print("Intialized tbro node")

    rclpy.spin(tbro_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    tbro_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
