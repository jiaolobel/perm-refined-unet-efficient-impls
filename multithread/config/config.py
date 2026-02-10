"""
!!! Attention: The input and output paths should be checked carefully. 

Experimental configuration.

Experiments
    0. Trick
        0.1 version = "original"
        0.2 version = "noif"
        0.3 version = "norm"
        0.4 version = "noeigen"
    1. (Skipped) Hyperparameter test, with grid search.
        1.1 theta_beta = 0.03125 with theta_alpha = 120.0 (4), 80.0 (1), 40.0 (2), 10.0 (3).
        1.2 theta_alpha = 80.0 with theta_beta = 0.03125 (1), 0.0625 (2), 0.125 (3), 0.25 (4).
    2. Exploration of multi-spectral features.
        2.1 self.img_channel_list, self.vis_channel_list = [4, 3, 2], None # RGB, acquired.
        2.2 self.img_channel_list, self.vis_channel_list = list(range(self.n_bands)), [4, 3, 2] # Seven-band.
    3. Dataset.
        3.1 L8

Default:
    theta_beta = 0.03125, theta_alpha = 80, channels = rgb, dataset = L8
"""

import sys
import importlib


class Config:
    def __init__(self) -> None:
        """
        In and Out paths should be double-checked.
        """
        # ==> version codes for module name and output, CRITICAL
        self.version = [
            "original",
            "noif",
            "norm",
            "noeigen",
            "hybrid",
            "hybridv2",
            "multiprocessingv2",
            "cppthread",
            "cppthreadv2",
            "cppthreadv3",
            "cppthreadv4",
            "threadpool",
        ][11]
        self.n_thread = [2, 4, 8, 16, 32][2]
        self.branch = "iiki2025"

        # ==> Hyperparameters, v1
        # self.theta_alpha, self.theta_beta, self.theta_gamma = 80.0, 0.03125, 3.0 # critical hyperparameters
        # ==> setting v2, for grid search
        self.theta_alpha = [80.0, 40.0, 10.0, 120.0][
            0
        ]  # 80.0, 40.0, 10.0, 120.0 with default theta_alpha=0.03125
        self.theta_beta = [0.03125, 0.0625, 0.125, 0.25][
            0
        ]  # 0.03125, 0.0625, 0.125, 0.25 with default theta_alpha=80.0
        self.theta_gamma = 3.0

        # ==> v1
        # self.img_channel_list, self.vis_channel_list = [4, 3, 2], None # bilateral features for CRF, RGB now
        # ==> setting v2
        self.channels = {
            "rgb": 0,
            "multispectral": 1,
        }["rgb"]

        # ==> dataset
        self.dataset = ["l8"][0]

        self.img_channel_list = [[4, 3, 2], [0, 1, 2, 3, 4, 5, 6]][self.channels]
        self.vis_channel_list = [None, [4, 3, 2]][self.channels]

        # !!! == Input and output, must be checked ==
        self.data_path = "E:/Research/experiment_data/{}/testcase".format(self.dataset)
        # self.save_path = "../output/a={}, b={}, r={}".format(self.theta_alpha, self.theta_beta, self.theta_gamma) # Perm. RFN. UNet
        self.save_path = "E:/Research/experiment_results/efficient_glob_perm_rfn_unet/{}/{}/{}/{}/{}/a={}, b={}, r={}".format(
            self.branch,
            self.version,
            self.dataset,
            self.channels,
            self.n_thread,
            self.theta_alpha,
            self.theta_beta,
            self.theta_gamma,
        )  # Perm. RFN. UNet w/o bilateral message-passing step

        # For code validate
        # self.save_path = "./{}/{}/{}/{}/{}/a={}, b={}, r={}".format(
        #     self.branch,
        #     self.version,
        #     self.dataset,
        #     self.channels,
        #     self.n_thread,
        #     self.theta_alpha,
        #     self.theta_beta,
        #     self.theta_gamma,
        # )
        self.save_info_fname = "log.csv"

        # # ==> repeat number for test
        # self.n_repeat = 3

        # - Model parameters
        self.n_bands = 7  # number of inputs of UNet
        self.n_classes = 4  # number of classes of UNet output
        self.model_path = "unary_model/"
        self.crop_height = 512
        self.crop_width = 512
        self.bilateral_compat = 10.0
        self.spatial_compat = 3.0
        self.n_iterations = 10

        sys.path.append("..")
        self.module = importlib.import_module(
            "model.{}.pydensecrf".format(self.version)
        )
        # self.dcrf = self.module.PyDenseCRF(
        #     H=self.height,
        #     W=self.width,
        #     n_classes=self.n_classes,
        #     d_bifeats=self.d_bifeats,
        #     d_spfeats=self.d_spfeats,
        #     theta_alpha=self.theta_alpha,
        #     theta_beta=self.theta_beta,
        #     theta_gamma=self.theta_gamma,
        #     bilateral_compat=self.bilateral_compat,
        #     spatial_compat=self.spatial_compat,
        #     n_iterations=self.n_iterations,
        #     n_thread=self.n_thread,
        # )
