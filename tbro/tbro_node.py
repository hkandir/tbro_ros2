import rclpy
from rclpy.node import Node

# replace this with dca1000_device/msg/MimoMsg
from std_msgs.msg import String
from dca1000_device.msg import MimoMsg


class TbroSubscriber(Node):

    def __init__(self):
        super().__init__("tbro_subscriber")
        self.controller_period = 0.1
        self.init_flag = True
        self.heatmap_buffer = []
        self.heatmap_cpi_first = None
        self.heatmap_cpi_second = None

        # heatmap subscriber
        self.heatmap_subcriber = self.create_subscription(
            MimoMsg, "/cascade/heatmap", self.listener_callback, 10
        )
        self.heatmap_subcriber  # prevent unused variable warning
        # Timer callback
        self.timer = self.create_timer(self.controller_period, self.timer_callback)

    def listener_callback(self, msg):
        self.get_logger().debug('I got heatmap msg: "%s"' % msg)
        self.heatmap_buffer.append(msg)
        # inti the system when first ever message is received
        if self.init_flag:
            self.heatmap_cpi_first = msg
            self.init_flag = False

    # Timer callback function
    def timer_callback(self):
        # print("Timer")

        self.get_logger().debug("Running tbro...")
        self.run_once()

    def process_frames(self, first_frame, second_frame):
        self.get_logger().info("Processing frames")

    def run_once(self):
        if self.heatmap_buffer:
            self.get_logger().debug("processng frames...")
            self.heatmap_cpi_second = self.heatmap_buffer.pop(0)
            self.process_frames(self.heatmap_cpi_first, self.heatmap_cpi_second)
            self.heatmap_cpi_first = self.heatmap_cpi_second


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
