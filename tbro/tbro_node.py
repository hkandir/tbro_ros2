"""
Transformer Based Radar Odometry ROS node
"""

# generic
import rclpy
from rclpy.node import Node
import math
import numpy as np
from tf_transformations import quaternion_from_euler

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
        self.controller_period = 0.01
        self.init_flag = False  # False if not initialzied
        self.msg_buffer = []
        self.data_set = MimoDataset()
        self.tbro_odometrty = np.zeros(6)
        self.args = Parameters()
        self.device = torch.device("cpu")
        # self.model = DeepROEncOnly(self.args)
        self.model = KramerOriginal(
            self.args,
            "//home/parallels/radar/models/epoch_25_batch_16_lr_1e-05_tbro_test_batch.model",
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

    def process_data(self):
        self.get_logger().debug("Processing frames")
        # self.get_logger().info("Zero (size): {}".format(self.data_set.__getitem__(0)))
        # self.get_logger().info("One: (size): {}".format(self.data_set.__getitem__(1)))
        # tmp = [self.data_set.__getitem__(0)[0], self.data_set.__getitem__(1)[0]]
        # print("tmp: {}".format(tmp))
        # print("tmp.len: ", tmp.__len__())

        with torch.no_grad():
            odom = self.model.forward(
                [self.data_set.__getitem__(0)[0], self.data_set.__getitem__(1)[0]]
            )

        self.tbro_odometrty[0] += float(odom[0][0])
        self.tbro_odometrty[1] += float(odom[0][1])
        self.tbro_odometrty[2] += float(odom[0][2])
        self.tbro_odometrty[3] += float(odom[0][3])
        self.tbro_odometrty[4] += float(odom[0][4])
        self.tbro_odometrty[5] += float(odom[0][5])

        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "world"
        odom_msg.child_frame_id = "imu_link_enu"
        odom_msg.pose.pose.position.x = self.tbro_odometrty[0]
        odom_msg.pose.pose.position.y = self.tbro_odometrty[1]
        odom_msg.pose.pose.position.z = self.tbro_odometrty[2]

        quaternion_frame = quaternion_from_euler(
            float(odom[0][3]), float(odom[0][4]), float(odom[0][5])
        )

        odom_msg.pose.pose.orientation.x = quaternion_frame[0]
        odom_msg.pose.pose.orientation.y = quaternion_frame[1]
        odom_msg.pose.pose.orientation.z = quaternion_frame[2]
        odom_msg.pose.pose.orientation.w = quaternion_frame[3]

        print("Publishing pose: ", odom_msg.pose.pose)

        self.odometry_publisher.publish(odom_msg)
        # print("image type: {}".format(type(self.data_set.__getitem__(0)[0])))

        # loader = DataLoader(
        #     self.data_set,
        #     batch_size=self.args.batch_size,
        #     shuffle=True,
        #     num_workers=4,
        #     drop_last=False,
        # )

        # for i, data in enumerate(loader):
        #     self.get_logger().info("index, data: {},{}".format(i, data))

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
