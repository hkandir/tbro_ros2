import rclpy
from rclpy.node import Node

# replace this with dca1000_device/msg/MimoMsg
from std_msgs.msg import String
from dca1000_device.msg import MimoMsg

class TbroSubscriber(Node):

    def __init__(self):
        super().__init__('tbro_subscriber')
        self.subscription = self.create_subscription(
            MimoMsg,
            '/cascade/heatmap',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg)



def main(args=None):
    # print('Hi from tbro.')
    rclpy.init(args=args)

    tbro_subscriber = TbroSubscriber()

    rclpy.spin(tbro_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    tbro_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
