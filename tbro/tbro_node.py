"""
Transformer Based Radar Odometry ROS node
"""

# generic
import rclpy
from rclpy.node import Node

# replace this with dca1000_device/msg/MimoMsg
# ros messages
from std_msgs.msg import String
from dca1000_device.msg import MimoMsg

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
        self.get_logger().info("Processing frames")
        # self.get_logger().info("Zero (size): {}".format(self.data_set.__getitem__(0)))
        # self.get_logger().info("One: (size): {}".format(self.data_set.__getitem__(1)))
        tmp = [self.data_set.__getitem__(0)[0], self.data_set.__getitem__(1)[0]]
        # print("tmp: {}".format(tmp))
        # print("tmp.len: ", tmp.__len__())

        with torch.no_grad():
            odom = self.model.forward(
                [self.data_set.__getitem__(0)[0], self.data_set.__getitem__(1)[0]]
            )

        print("odom: ", odom)

        # TODO: publish odom here

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
