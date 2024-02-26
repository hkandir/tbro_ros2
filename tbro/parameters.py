import os


class Parameters:
    def __init__(self):

        # Root directory in which the ColoRadar Bags are stored
        # self.directory = '/media/giantdrive/andrew/ColoRadar/resampled/'
        self.directory = "/home/kharlow/andrew/resampled/"
        # self.directory = '/home/kharlow/andrew/velocity_resampled/'
        # self.directory = '/home/arpg/datasets/andrew/resampled/'
        # Text file containing bag file names
        self.list_bags = "bag_list.txt"
        # Radar topic, type dca1000_device/RadarCubeMsg
        self.topic_radar = "/cascade/heatmap"
        # Odometry Topic used to align heatmaps, type nav_msgs/Odometry
        self.topic_odom = "/lidar_ground_truth"
        # File containing odom to radar transform
        self.tf_base_to_radar = "calib/transforms/base_to_cascade.txt"
        # File containing pre-trained encoder weights
        # self.pretrained_enc_path = '/home/arpg/results_training/results/best_encoder_modified.model'
        # self.pretrained_enc_path = '/home/kharlow/ws/TBRO/src/TBRO/scripts/models/weights/best_encoder_modified.model'
        self.pretrained_enc_path = None

        self.forward_seq_only = False

        # Max sequence length (optional) (int or None for full sequence)
        self.max_length = 7
        self.batch_size = 2
        self.epochs = 50

        self.radar_shape = [64, 128, 64]
        self.mean_enable = True
        self.hidden_size = 1000

        # self.learning_rate = 1.0e-7
        # self.learning_rate = 1.0e-5
        self.learning_rate = 4.0e-3

        self.alphas = [1.0, 1.0]
        self.betas = [1.0, 1.0]
        self.gammas = [1.0, 1.0]

        # Text file with filesnames from which to load training data
        self.train_files = "med_train_dataset.txt"
        # Text file with filenames from which to load validation data
        self.val_files = "med_test_dataset.txt"
        # File in which to save the model weight
        self.save_filename = (
            "epoch_"
            + str(self.epochs)
            + "_batch_"
            + str(self.batch_size)
            + "_lr_"
            + str(self.learning_rate)
            + "_pretrained_tbro_test_batch.model"
        )
        # File from which to load model for testing only
        self.load_filename = ""
        # Bool, load model and run on test data only without training
        self.eval_only = False

        # TEST VARIABLES:
