import rclpy
from rclpy.node import Node

# replace this with dca1000_device/msg/MimoMsg
from std_msgs.msg import String
from dca1000_device.msg import MimoMsg

# Custom dataset for TBRO
# from mimo_dataset import MimoDataset
import torch
from torch.utils.data import Dataset


class MimoDataset(Dataset):
    def __init__(self):
        super(MimoDataset, self).__init__()
        self.image_list = []

    def load_img(self, new_image):
        if len(self.image_list) > 1:
            self.image_list.pop(0)
        self.image_list.append(new_image)

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, index):
        image = torch.FloatTensor(self.image_list[index])
        lable = ...

        return image, lable


class TbroSubscriber(Node):

    def __init__(self):
        super().__init__("tbro_subscriber")
        self.controller_period = 0.1
        self.init_flag = False  # False if not initialzied
        self.heatmap_buffer = []

        # heatmap subscriber
        self.heatmap_subcriber = self.create_subscription(
            MimoMsg, "/cascade/heatmap", self.listener_callback, 10
        )
        self.heatmap_subcriber  # prevent unused variable warning
        # Timer callback
        self.timer = self.create_timer(self.controller_period, self.timer_callback)

        self.data_set = MimoDataset()

    def listener_callback(self, msg):
        self.get_logger().debug('I got heatmap msg: "%s"' % msg)
        # inti the system when first ever message is received
        if not self.init_flag:
            self.data_set.load_img(msg.image)
            if self.data_set.__len__() == 2:
                self.init_flag = True
        else:
            self.heatmap_buffer.append(msg.image)

    # Timer callback function
    def timer_callback(self):
        # print("Timer")

        self.get_logger().debug("Running tbro...")
        self.run_once()

    def process_data(self):
        self.get_logger().info("Processing frames")
        self.get_logger().info("Zero: ".format(self.data_set.__getitem__(0)))
        self.get_logger().info("One:  ".format(self.data_set.__getitem__(1)))

    def run_once(self):
        if self.init_flag:
            # TODO: This will miss the first message. Fix it.
            if len(self.heatmap_buffer) > 0:
                self.data_set.load_img(self.heatmap_buffer.pop(0))
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
