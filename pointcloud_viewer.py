__author__ = "Martin Hahner"
__contact__ = "martin.hahner@pm.me"
__license__ = "CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)"

# GUI adapted from
# https://memotut.com/create-a-3d-model-viewer-with-pyqt5-and-pyqtgraph-b3916/ and
# https://matplotlib.org/3.1.1/gallery/user_interfaces/embedding_in_qt_sgskip.html

import multiprocessing as mp
if __name__ == '__main__':
    mp.set_start_method("spawn")

import os
import sys
import copy
import gzip
import socket
import pandas
import logging
import argparse
import tempfile
import warnings

warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import numpy as np
import pickle as pkl
import matplotlib as mpl
import matplotlib.cm as cm
import pyqtgraph.opengl as gl

from glob import glob
from pathlib import Path
from pprint import pprint
from plyfile import PlyData
from typing import List, Dict

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from pyqtgraph.Qt import QtGui

from lib.LISA.python.lisa import LISA

from lib.cadc_devkit.other.create_image_sets import DROR_LEVELS

from lib.OpenPCDet.pcdet.utils import calibration_kitti


from lib.LiDAR_fog_sim.fog_simulation import ParameterSet, simulate_fog
from lib.LiDAR_fog_sim.SeeingThroughFog.tools.DatasetViewer.lib.read import load_calib_data, read_label
from lib.LiDAR_fog_sim.SeeingThroughFog.tools.DatasetFoggification.beta_modification import BetaRadomization
from lib.LiDAR_fog_sim.SeeingThroughFog.tools.DatasetFoggification.lidar_foggification import haze_point_cloud

from tools.wet_ground.augmentation import ground_water_augmentation

from tools.snowfall.simulation import augment
from tools.snowfall.sampling import snowfall_rate_to_rainfall_rate, compute_occupancy

try:

    import torch

    CUDA_AVAILABLE = torch.cuda.is_available()

    from pcdet.utils import common_utils
    from pcdet.datasets import build_dataloader
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.models import build_network, load_data_to_gpu

except ImportError as e:

    CUDA_AVAILABLE = False

    if __name__ == '__main__':
        print(f'{e} => Inference not available')

try:

    from lib.cadc_devkit.other.dror import dynamic_radius_outlier_filter, get_cube_mask

    live_DROR_available = True

except ImportError as e:

    if __name__ == '__main__':
        print(f'{e} => live DROR not available')

    live_DROR_available = False


    def get_cube_mask() -> None:
        raise NotImplementedError


    def dynamic_radius_outlier_filter(sensor: str, signal: str, variant: str, before: int, filename: str) -> np.ndarray:

        path = f'{DROR}/alpha_0.45/all/{sensor}/{signal}/{variant}'
        name = Path(filename).name.replace('.bin', '')
        file = f'{path}/{name}.pkl'

        with open(file, 'rb') as f:
            snow_indices = pkl.load(f)

        mask = np.ones(before, dtype=bool)

        mask[snow_indices] = False

        return mask

MIN_DIST = 3  # in m -> to hide "the ring" around the sensor

CAMERA_READY = True
SAVE_IMAGES = False
ROTATE = 0

BLACK = (  0,   0,   0, 255)
WHITE = (255, 255, 255, 255)

        #    R,   G,   B, alpha
COLORS = [(  0, 255,   0, 255), # cars in green
          (255,   0,   0, 255), # pedestrian in red
          (255, 255,   0, 255)] # cyclists in yellow

if CAMERA_READY:
    DET_COLORS = [BLACK,  # cars
                  BLACK,  # pedestrian
                  BLACK]  # cyclists

    GRAY = (0, 0, 0, 100)

else:
    #                R,   G,   B, alpha
    DET_COLORS = [(  0, 255,   0, 100), # cars in green
                  (255,   0,   0, 100), # pedestrian in red
                  (255, 255,   0, 100)] # cyclists in yellow

    GRAY =        (255, 255, 255, 100)


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--datasets', type=str, help='path to where you store your datasets',
                    default=str(Path.home() / 'datasets'))
parser.add_argument('-e', '--experiments', type=str, help='path to where you store your OpenPCDet experiments',
                    default=str(Path(__file__).parent.absolute() / 'experiments'))
args = parser.parse_args()

DATASETS_ROOT = Path(args.datasets)
EXPERIMENTS_ROOT = Path(args.experiments)

DROR = Path(__file__).parent.absolute()

AUDI = DATASETS_ROOT / 'A2D2/camera_lidar_semantic_bboxes'
LYFT = DATASETS_ROOT / 'LyftLevel5/Perception/train_lidar'
ARGO = DATASETS_ROOT / 'Argoverse'
PANDA = DATASETS_ROOT / 'PandaSet'
DENSE = DATASETS_ROOT / 'DENSE/SeeingThroughFog/lidar_hdl64_strongest'
KITTI = DATASETS_ROOT / 'KITTI/3D/training/velodyne'
WAYMO = DATASETS_ROOT / 'WaymoOpenDataset/WOD/train/velodyne'
HONDA = DATASETS_ROOT / 'Honda_3D/scenarios'
APOLLO = DATASETS_ROOT / 'Apollo3D'
NUSCENES = DATASETS_ROOT / 'nuScenes/sweeps/LIDAR_TOP'

HOSTNAME = socket.gethostname()

if HOSTNAME == 'beauty' or HOSTNAME == 'beast':
    DENSE = Path.home() / 'datasets_local' / 'DENSE/SeeingThroughFog/lidar_hdl64_strongest'


def get_calib(sensor: str = 'hdl64'):
    calib_file = Path(__file__).parent.absolute() / 'lib' / 'OpenPCDet' / 'data' / 'dense' / f'calib_{sensor}.txt'
    assert calib_file.exists(), f'{calib_file} not found'
    return calibration_kitti.Calibration(calib_file)


def get_fov_flag(pts_rect, img_shape, calib):
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

    return pts_valid_flag


def close_all_windows(exit_code: int = 0) -> None:
    sys.exit(exit_code)


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class QHLine(QFrame):

    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)


class QVLine(QFrame):

    def __init__(self):
        super(QVLine, self).__init__()
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)


class ImageWindow(QMainWindow):

    def __init__(self) -> None:

        super(ImageWindow, self).__init__()

        self.monitor = QDesktopWidget().screenGeometry(0)

        self.setGeometry(self.monitor)

        self.centerWidget = QWidget()
        self.setCentralWidget(self.centerWidget)

        self.layout = QGridLayout()
        self.centerWidget.setLayout(self.layout)

        self.image_label = QLabel()
        self.layout.addWidget(self.image_label, 0, 0)


# noinspection PyUnresolvedReferences
class LidarWindow(QMainWindow):

    def __init__(self) -> None:

        super(LidarWindow, self).__init__()

        self.show_img = False
        self.show_temp = False
        self.image_window = ImageWindow()

        self.boxes = {}
        self.label = None
        self.model = None
        self.config = None
        self.logger = None
        self.sampler = None
        self.eval_set = None
        self.eval_loader = None
        self.predictions = None
        self.prediction_boxes = []
        self.result_dict = {}
        self.show_predictions = False
        self.prediction_threshold = 50

        self.gain = True
        self.noise_variant = 'v4'

        self.noise = 10

        self.min_fog_response = -1
        self.max_fog_response = -1
        self.num_fog_responses = -1

        self.dror_alpha = 0.45  # HDL64 Spec Sheet: Angular Resolution (Horizontal/Azimuth): 0.08° – 0.35°
        self.dror_beta = 3
        self.dror_k_min = 3
        self.dror_sr_min = 4  # in cm

        self.dror_alpha_scale = 100

        self.p = ParameterSet(gamma=0.000001,
                              gamma_min=0.0000001,
                              gamma_max=0.00001,
                              gamma_scale=10000000)

        self.p.beta_0 = self.p.gamma / np.pi
        self.row_height = 20

        self.monitor = QDesktopWidget().screenGeometry(0)

        self.setGeometry(self.monitor)

        self.setAcceptDrops(True)

        self.simulated_fog = False
        self.simulated_fog_pc = None

        self.simulated_fog_dense = False

        self.color_dict = {0: 'x',
                           1: 'y',
                           2: 'z',
                           3: 'intensity',
                           4: 'distance',
                           5: 'angle',
                           6: 'channel'}

        self.min_value = 0
        self.max_value = 63
        self.point_size = 3
        self.line_width = 3

        if CAMERA_READY:
            self.color_feature = 3
        else:
            self.color_feature = 2

        self.threshold = 50
        self.num_features = 5
        self.dataset = None
        self.success = False
        self.extension = 'bin'
        self.d_type = np.float32
        self.intensity_multiplier = 1
        self.color_name = self.color_dict[self.color_feature]

        self.min_height = -400  # in cm
        self.max_distance = 80  # in m

        self.lastDir = None
        self.droppedFilename = None

        self.current_pc = None
        self.current_mesh = None

        self.temporal_pcs = []
        self.temporal_meshes = []

        self.file_name = None
        self.file_list = None
        self.index = -1

        self.centerWidget = QWidget()
        self.setCentralWidget(self.centerWidget)

        self.layout = QGridLayout()
        self.centerWidget.setLayout(self.layout)

        self.num_columns = 6
        self.current_row = 0

        self.viewer = gl.GLViewWidget()
        if CAMERA_READY:
            self.viewer.setBackgroundColor('w')
        self.viewer.setWindowTitle('drag & drop point cloud viewer')
        self.viewer.opts['center'] = QVector3D(15, 0, 0)
        self.viewer.opts['distance'] = 30
        self.viewer.opts['elevation'] = 5
        self.viewer.opts['azimuth'] = 180

        self.layout.addWidget(self.viewer, 0, 6, 30, 170)

        self.show_dror_cube = False
        try:
            self.dror_cube = gl.GLBoxItem(QtGui.QVector3D(1, 1, 1), color=GRAY, line_width=self.line_width)
        except TypeError:
            self.dror_cube = gl.GLBoxItem(QtGui.QVector3D(1, 1, 1), color=GRAY)
        self.dror_cube.setSize(10, 2, 2)
        self.dror_cube.translate(3, -1, -1)

        self.current_row += 0

        self.num_info = QLabel("")
        self.num_info.setAlignment(Qt.AlignCenter)
        self.num_info.setMaximumSize(self.monitor.width(), self.row_height)
        self.layout.addWidget(self.num_info, self.current_row, 0)

        self.file_name_label = QLabel()
        self.file_name_label.setAlignment(Qt.AlignCenter)
        self.file_name_label.setMaximumSize(self.monitor.width(), 20)
        self.layout.addWidget(self.file_name_label, self.current_row, 1)

        self.inference_btn = QPushButton('run inference')
        self.inference_btn.setEnabled(False)
        self.inference_btn.clicked.connect(self.run_inference)
        self.layout.addWidget(self.inference_btn, self.current_row, 3, 1, 2)

        self.minus6 = QCheckBox("-6")
        self.minus6.setEnabled(False)
        self.minus6.stateChanged.connect(self.update_temporal_clouds)
        self.layout.addWidget(self.minus6, self.current_row, 5)

        self.current_row += 1

        self.choose_dir_btn = QPushButton("choose custom directory")
        self.choose_dir_btn.clicked.connect(self.show_directory_dialog)
        self.layout.addWidget(self.choose_dir_btn, self.current_row, 1)

        self.prev_btn = QPushButton("<-")
        self.next_btn = QPushButton("->")

        self.prev_btn.clicked.connect(self.decrement_index)
        self.next_btn.clicked.connect(self.increment_index)

        self.layout.addWidget(self.prev_btn, self.current_row, 0)
        self.layout.addWidget(self.next_btn, self.current_row, 2)

        self.load_kitti_btn = QPushButton("KITTI")
        self.load_kitti_btn.clicked.connect(self.load_kitti)
        self.layout.addWidget(self.load_kitti_btn, self.current_row, 3)

        self.load_nuscenes_btn = QPushButton("nuScenes")
        self.load_nuscenes_btn.clicked.connect(self.load_nuscenes)
        self.layout.addWidget(self.load_nuscenes_btn, self.current_row, 4)

        self.minus5 = QCheckBox("-5")
        self.minus5.setEnabled(False)
        self.minus5.stateChanged.connect(self.update_temporal_clouds)
        self.layout.addWidget(self.minus5, self.current_row, 5)

        self.current_row += 1

        if self.show_predictions:
            self.visualize_predictions_btn = QPushButton('hide predictions', self)
        else:
            self.visualize_predictions_btn = QPushButton('show predictions', self)

        self.visualize_predictions_btn.setEnabled(False)
        self.layout.addWidget(self.visualize_predictions_btn, self.current_row, 0)
        self.visualize_predictions_btn.clicked.connect(self.toggle_predictions)

        self.experiment_path_box = QLineEdit(self)
        self.experiment_path_box.setText('snow+wet')
        self.layout.addWidget(self.experiment_path_box, self.current_row, 1)
        self.current_experiment = self.experiment_path_box.text()

        self.load_experiment_path_btn = QPushButton('load results', self)
        self.layout.addWidget(self.load_experiment_path_btn, self.current_row, 2)
        self.load_experiment_path_btn.clicked.connect(self.load_results)

        self.load_dense_btn = QPushButton("DENSE")
        self.load_dense_btn.clicked.connect(self.load_dense)
        self.layout.addWidget(self.load_dense_btn, self.current_row, 3)

        self.load_lyft_btn = QPushButton("LyftL5")
        self.load_lyft_btn.clicked.connect(self.load_lyft)
        self.layout.addWidget(self.load_lyft_btn, self.current_row, 4)

        self.minus4 = QCheckBox("-4")
        self.minus4.setEnabled(False)
        self.minus4.stateChanged.connect(self.update_temporal_clouds)
        self.layout.addWidget(self.minus4, self.current_row, 5)

        self.current_row += 1

        self.prediction_threshold_title = QLabel("prediction confidence")
        self.prediction_threshold_title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.prediction_threshold_title, self.current_row, 0)

        self.prediction_threshold_label = QLabel(str(self.prediction_threshold))
        self.prediction_threshold_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.prediction_threshold_label, self.current_row, 2)

        self.prediction_threshold_slider = QSlider(Qt.Horizontal)
        self.prediction_threshold_slider.setMinimum(0)
        self.prediction_threshold_slider.setMaximum(100)
        self.prediction_threshold_slider.setValue(self.prediction_threshold)
        self.prediction_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.prediction_threshold_slider.setTickInterval(10)
        self.prediction_threshold_slider.setEnabled(False)

        self.layout.addWidget(self.prediction_threshold_slider, self.current_row, 1)
        self.prediction_threshold_slider.valueChanged.connect(self.prediction_threshold_slider_change)

        self.load_honda_btn = QPushButton("H3D")
        self.load_honda_btn.clicked.connect(self.load_honda)
        self.layout.addWidget(self.load_honda_btn, self.current_row, 3)

        self.load_argo_btn = QPushButton("Argoverse")
        self.load_argo_btn.clicked.connect(self.load_argo)
        self.layout.addWidget(self.load_argo_btn, self.current_row, 4)

        self.minus3 = QCheckBox("-3")
        self.minus3.setEnabled(False)
        self.minus3.stateChanged.connect(self.update_temporal_clouds)
        self.layout.addWidget(self.minus3, self.current_row, 5)

        self.current_row += 1

        self.mor_label = QLabel(f'meteorological optical range (MOR) = {round(self.p.mor, 2)}m')
        self.mor_label.setAlignment(Qt.AlignCenter)
        self.mor_label.setMaximumSize(self.monitor.width(), self.row_height)
        self.layout.addWidget(self.mor_label, self.current_row, 1)

        self.load_audi_btn = QPushButton("A2D2")
        self.load_audi_btn.clicked.connect(self.load_audi)
        self.layout.addWidget(self.load_audi_btn, self.current_row, 3)

        self.load_waymo_btn = QPushButton("Waymo")
        self.load_waymo_btn.clicked.connect(self.load_waymo)
        self.layout.addWidget(self.load_waymo_btn, self.current_row, 4)

        self.minus2 = QCheckBox("-2")
        self.minus2.setEnabled(False)
        self.minus2.stateChanged.connect(self.update_temporal_clouds)
        self.layout.addWidget(self.minus2, self.current_row, 5)

        self.current_row += 1

        self.alpha_title = QLabel('attenuation coefficient')
        self.alpha_title.setAlignment(Qt.AlignRight)
        self.layout.addWidget(self.alpha_title, self.current_row, 0)

        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setMinimum(int(self.p.alpha_min * self.p.alpha_scale))
        self.alpha_slider.setMaximum(int(self.p.alpha_max * self.p.alpha_scale))
        self.alpha_slider.setValue(int(self.p.alpha * self.p.alpha_scale))

        self.layout.addWidget(self.alpha_slider, self.current_row, 1)
        self.alpha_slider.valueChanged.connect(self.update_labels)

        self.alpha_label = QLabel(f"\u03B1 = {self.p.alpha}")
        self.alpha_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.alpha_label, self.current_row, 2)

        self.load_panda_btn = QPushButton("PandaSet")
        self.load_panda_btn.clicked.connect(self.load_panda)
        self.layout.addWidget(self.load_panda_btn, self.current_row, 3)

        self.load_apollo_btn = QPushButton("Apollo")
        self.load_apollo_btn.clicked.connect(self.load_apollo)
        self.layout.addWidget(self.load_apollo_btn, self.current_row, 4)

        self.minus1 = QCheckBox("-1")
        self.minus1.setEnabled(False)
        self.minus1.stateChanged.connect(self.update_temporal_clouds)
        self.layout.addWidget(self.minus1, self.current_row, 5)

        self.current_row += 1

        self.beta_title = QLabel('backscattering coefficient')
        self.beta_title.setAlignment(Qt.AlignRight)
        self.layout.addWidget(self.beta_title, self.current_row, 0)

        self.beta_slider = QSlider(Qt.Horizontal)
        self.beta_slider.setMinimum(int(self.p.beta_min * self.p.beta_scale))
        self.beta_slider.setMaximum(int(self.p.beta_max * self.p.beta_scale))
        self.beta_slider.setValue(int(self.p.beta * self.p.beta_scale))

        self.layout.addWidget(self.beta_slider, self.current_row, 1)
        self.beta_slider.valueChanged.connect(self.update_labels)

        self.beta_label = QLabel(f"\u03B2 = {round(self.p.beta * self.p.mor, 3)} / MOR")
        self.beta_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.beta_label, self.current_row, 2)

        self.toggle_img_btn = QPushButton("show image")
        self.toggle_img_btn.setEnabled(False)
        self.toggle_img_btn.clicked.connect(self.toggle_image_visibility)
        self.layout.addWidget(self.toggle_img_btn, self.current_row, 3)

        self.toggle_temp_btn = QPushButton("show temporal")
        self.toggle_temp_btn.setEnabled(False)
        self.toggle_temp_btn.clicked.connect(self.toggle_temp_visibility)
        self.layout.addWidget(self.toggle_temp_btn, self.current_row, 4)

        self.zero = QCheckBox("0")
        self.zero.setEnabled(False)
        self.zero.setChecked(True)
        self.zero.stateChanged.connect(self.update_temporal_clouds)
        self.layout.addWidget(self.zero, self.current_row, 5)

        self.current_row += 1

        self.gamma_title = QLabel("reflextivity of the hard target")
        self.gamma_title.setAlignment(Qt.AlignRight)
        self.layout.addWidget(self.gamma_title, self.current_row, 0)

        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setMinimum(int(self.p.gamma_min * self.p.gamma_scale))
        self.gamma_slider.setMaximum(int(self.p.gamma_max * self.p.gamma_scale))
        self.gamma_slider.setValue(int(self.p.gamma * self.p.gamma_scale))

        self.layout.addWidget(self.gamma_slider, self.current_row, 1)
        self.gamma_slider.valueChanged.connect(self.update_labels)

        self.gamma_label = QLabel(f"\u0393 = {self.p.gamma}")
        self.gamma_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.gamma_label, self.current_row, 2)

        self.show_fov_only = True
        self.show_fov_only_btn = QPushButton('show full 360°' if self.show_fov_only else 'show camera FOV only')
        self.show_fov_only_btn.setEnabled(False)
        self.show_fov_only_btn.clicked.connect(self.toggle_fov)
        self.layout.addWidget(self.show_fov_only_btn, self.current_row, 3, 1, 2)

        self.plus1 = QCheckBox("+1")
        self.plus1.setEnabled(False)
        self.plus1.stateChanged.connect(self.update_temporal_clouds)
        self.layout.addWidget(self.plus1, self.current_row, 5)

        self.current_row += 1

        self.plus2 = QCheckBox("+2")
        self.plus2.setEnabled(False)
        self.plus2.stateChanged.connect(self.update_temporal_clouds)
        self.layout.addWidget(self.plus2, self.current_row, 5)

        self.log_info = QLabel("")
        self.log_info.setAlignment(Qt.AlignCenter)
        self.log_info.setMaximumSize(self.monitor.width(), self.row_height)
        self.layout.addWidget(self.log_info, self.current_row, 1)

        self.dense_split_paths = []

        self.cb_splits = QComboBox()
        # self.cb_splits.setEditable(True)
        # self.cb_splits.lineEdit().setReadOnly(True)
        # self.cb_splits.lineEdit().setAlignment(Qt.AlignCenter)
        self.cb_splits.addItems(self.populate_dense_splits())
        self.cb_splits.currentIndexChanged.connect(self.split_selection_change)
        self.cb_splits.setEnabled(False)
        self.layout.addWidget(self.cb_splits, self.current_row, 3, 1, 2)

        self.current_row += 1

        self.color_title = QLabel("color code")
        self.color_title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.color_title, self.current_row, 0)

        self.color_label = QLabel(self.color_name)
        self.color_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.color_label, self.current_row, 2)

        self.color_slider = QSlider(Qt.Horizontal)
        self.color_slider.setMinimum(0)
        self.color_slider.setMaximum(6)
        self.color_slider.setValue(self.color_feature)
        self.color_slider.setTickPosition(QSlider.TicksBelow)
        self.color_slider.setTickInterval(1)

        self.layout.addWidget(self.color_slider, self.current_row, 1)
        self.color_slider.valueChanged.connect(self.color_slider_change)

        self.sensor = 'hdl64'
        self.cb_sensors = QComboBox()
        self.cb_sensors.addItems(['hdl64', 'vlp32'])
        self.cb_sensors.currentIndexChanged.connect(self.sensor_selection_change)
        self.cb_sensors.setEnabled(False)
        self.layout.addWidget(self.cb_sensors, self.current_row, 3)

        self.signal = 'strongest'
        self.cb_signals = QComboBox()
        self.cb_signals.addItems(['strongest', 'last'])
        self.cb_signals.currentIndexChanged.connect(self.signal_selection_change)
        self.cb_signals.setEnabled(False)
        self.layout.addWidget(self.cb_signals, self.current_row, 4)

        self.plus3 = QCheckBox("+3")
        self.plus3.setEnabled(False)
        self.plus3.stateChanged.connect(self.update_temporal_clouds)
        self.layout.addWidget(self.plus3, self.current_row, 5)

        self.current_row += 1

        self.min_height_title = QLabel("minimum height")
        self.min_height_title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.min_height_title, self.current_row, 0)

        self.min_height_label = QLabel(f'{self.min_height} cm')
        self.min_height_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.min_height_label, self.current_row, 2)

        self.min_height_slider = QSlider(Qt.Horizontal)
        self.min_height_slider.setMinimum(-400)
        self.min_height_slider.setMaximum(0)
        self.min_height_slider.setValue(self.min_height)
        self.min_height_slider.setTickPosition(QSlider.TicksBelow)
        self.min_height_slider.setTickInterval(10)

        self.layout.addWidget(self.min_height_slider, self.current_row, 1)
        self.min_height_slider.valueChanged.connect(self.min_height_slider_change)

        if self.simulated_fog:
            self.alpha_slider.setEnabled(True)
            self.beta_slider.setEnabled(True)
            self.gamma_slider.setEnabled(True)
            self.toggle_simulated_fog_btn = QPushButton("remove our fog")
        else:
            self.alpha_slider.setEnabled(False)
            self.beta_slider.setEnabled(False)
            self.gamma_slider.setEnabled(False)
            self.toggle_simulated_fog_btn = QPushButton("add our fog")

        self.toggle_simulated_fog_btn.clicked.connect(self.toggle_simulated_fog)
        self.layout.addWidget(self.toggle_simulated_fog_btn, self.current_row, 3)

        self.toggle_simulated_fog_btn.setEnabled(False)

        if self.simulated_fog_dense:
            self.toggle_simulated_fog_dense_btn = QPushButton("remove STF fog")
        else:
            self.toggle_simulated_fog_dense_btn = QPushButton("add STF fog")

        self.toggle_simulated_fog_dense_btn.clicked.connect(self.toggle_simulated_fog_dense)
        self.layout.addWidget(self.toggle_simulated_fog_dense_btn, self.current_row, 4)
        self.toggle_simulated_fog_dense_btn.setEnabled(False)

        self.plus4 = QCheckBox("+4")
        self.plus4.setEnabled(False)
        self.plus4.stateChanged.connect(self.update_temporal_clouds)
        self.layout.addWidget(self.plus4, self.current_row, 5)

        self.current_row += 1

        self.max_distance_title = QLabel("maximum distance")
        self.max_distance_title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.max_distance_title, self.current_row, 0)

        self.max_distance_label = QLabel(f'{self.max_distance} m')
        self.max_distance_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.max_distance_label, self.current_row, 2)

        self.max_distance_slider = QSlider(Qt.Horizontal)
        self.max_distance_slider.setMinimum(0)
        self.max_distance_slider.setMaximum(200)
        self.max_distance_slider.setValue(self.max_distance)
        self.max_distance_slider.setTickPosition(QSlider.TicksBelow)
        self.max_distance_slider.setTickInterval(10)

        self.layout.addWidget(self.max_distance_slider, self.current_row, 1)
        self.max_distance_slider.valueChanged.connect(self.max_distance_slider_change)

        self.current_row += 1

        self.layout.addWidget(QHLine(), self.current_row, 0, 1, self.num_columns)

        self.current_row += 1

        self.dror_intensity = 'DROR'

        self.dror_headline = QLabel(self.dror_intensity)
        self.dror_headline.setAlignment(Qt.AlignCenter)
        self.dror_headline.setMaximumSize(self.monitor.width(), 20)
        self.layout.addWidget(self.dror_headline, self.current_row, 1)

        self.apply_dror = False

        self.toggle_cube_btn = QPushButton('hide cube' if self.show_dror_cube else 'show_cube')
        self.toggle_cube_btn.setEnabled(False)
        self.toggle_cube_btn.clicked.connect(self.toggle_dror_cube)
        self.layout.addWidget(self.toggle_cube_btn, self.current_row, 3)

        self.toggle_dror_btn = QPushButton("apply DROR")
        self.toggle_dror_btn.clicked.connect(self.toggle_dror)
        self.layout.addWidget(self.toggle_dror_btn, self.current_row, 4)
        self.toggle_dror_btn.setEnabled(False)

        self.current_row += 1

        self.dror_alpha_title = QLabel("horizontal angular resolution")
        self.dror_alpha_title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.dror_alpha_title, self.current_row, 0)

        self.dror_alpha_label = QLabel(f"\u03B1 = {self.dror_alpha}")
        self.dror_alpha_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.dror_alpha_label, self.current_row, 2)

        self.dror_alpha_slider = QSlider(Qt.Horizontal)
        self.dror_alpha_slider.setMinimum(0)
        self.dror_alpha_slider.setMaximum(100)
        self.dror_alpha_slider.setValue(int(self.dror_alpha * self.dror_alpha_scale))
        self.dror_alpha_slider.setTickPosition(QSlider.TicksBelow)
        self.dror_alpha_slider.setTickInterval(10)
        self.dror_alpha_slider.setEnabled(False)

        self.layout.addWidget(self.dror_alpha_slider, self.current_row, 1)
        self.dror_alpha_slider.valueChanged.connect(self.dror_change)

        self.snowfall_rate_title = QLabel('snowfall rate')
        self.snowfall_rate_title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.snowfall_rate_title, self.current_row, 3)

        self.velocity_title = QLabel('velocity')
        self.velocity_title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.velocity_title, self.current_row, 4)

        self.current_row += 1

        self.dror_beta_title = QLabel("multiplication factor")
        self.dror_beta_title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.dror_beta_title, self.current_row, 0)

        self.dror_beta_label = QLabel(f"\u03B2 = {self.dror_beta}")
        self.dror_beta_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.dror_beta_label, self.current_row, 2)

        self.dror_beta_slider = QSlider(Qt.Horizontal)
        self.dror_beta_slider.setMinimum(0)
        self.dror_beta_slider.setMaximum(10)
        self.dror_beta_slider.setValue(int(self.dror_beta))
        self.dror_beta_slider.setTickPosition(QSlider.TicksBelow)
        self.dror_beta_slider.setTickInterval(1)
        self.dror_beta_slider.setEnabled(False)

        self.layout.addWidget(self.dror_beta_slider, self.current_row, 1)
        self.dror_beta_slider.valueChanged.connect(self.dror_change)

        self.cb_snowfall_rate = QComboBox()
        self.cb_snowfall_rate.addItems(['0.5', '1', '1.5', '2', '2.5'])
        self.cb_snowfall_rate.currentIndexChanged.connect(self.snowfall_change)
        self.cb_snowfall_rate.setEnabled(False)
        self.layout.addWidget(self.cb_snowfall_rate, self.current_row, 3)

        self.cb_velocity = QComboBox()
        self.cb_velocity.addItems(['0.2', '0.4', '0.6', '0.8', '1', '1.2', '1.4', '1.6', '1.8', '2'])
        self.cb_velocity.currentIndexChanged.connect(self.snowfall_change)
        self.cb_velocity.setEnabled(False)
        self.layout.addWidget(self.cb_velocity, self.current_row, 4)

        self.current_row += 1

        self.dror_k_min_title = QLabel("min. number of neighbors")
        self.dror_k_min_title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.dror_k_min_title, self.current_row, 0)

        self.dror_k_min_label = QLabel(f"k_min = {self.dror_k_min}")
        self.dror_k_min_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.dror_k_min_label, self.current_row, 2)

        self.dror_k_min_slider = QSlider(Qt.Horizontal)
        self.dror_k_min_slider.setMinimum(0)
        self.dror_k_min_slider.setMaximum(10)
        self.dror_k_min_slider.setValue(self.dror_k_min)
        self.dror_k_min_slider.setTickPosition(QSlider.TicksBelow)
        self.dror_k_min_slider.setTickInterval(1)
        self.dror_k_min_slider.setEnabled(False)

        self.layout.addWidget(self.dror_k_min_slider, self.current_row, 1)
        self.dror_k_min_slider.valueChanged.connect(self.dror_change)

        self.rainfall_rate_title = QLabel('')
        self.rainfall_rate_title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.rainfall_rate_title, self.current_row, 3, 1, 2)

        self.current_row += 1

        self.dror_sr_min_title = QLabel('min. search radius')
        self.dror_sr_min_title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.dror_sr_min_title, self.current_row, 0)

        self.dror_sr_min_label = QLabel(f'sr_min = {self.dror_sr_min}cm')
        self.dror_sr_min_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.dror_sr_min_label, self.current_row, 2)

        self.dror_sr_min_slider = QSlider(Qt.Horizontal)
        self.dror_sr_min_slider.setMinimum(0)
        self.dror_sr_min_slider.setMaximum(100)
        self.dror_sr_min_slider.setValue(self.dror_sr_min)
        self.dror_sr_min_slider.setTickPosition(QSlider.TicksBelow)
        self.dror_sr_min_slider.setTickInterval(10)
        self.dror_sr_min_slider.setEnabled(False)

        self.layout.addWidget(self.dror_sr_min_slider, self.current_row, 1)
        self.dror_sr_min_slider.valueChanged.connect(self.dror_change)

        self.toggle_snow_btn = QPushButton("apply snow")
        self.toggle_snow_btn.setEnabled(False)
        self.toggle_snow_btn.clicked.connect(self.toggle_snow)
        self.layout.addWidget(self.toggle_snow_btn, self.current_row, 4)

        self.current_row += 1

        self.layout.addWidget(QHLine(), self.current_row, 0, 1, self.num_columns)

        self.current_row += 1

        ########
        # SNOW #
        ########

        self.lisa = None
        self.tmp_value = None

        self.fixed_seed = True
        self.apply_wet = False
        self.apply_snow = False
        self.apply_lisa = False

        self.mode = 'gunn'
        self.rain_rate = 71  # (mm/hr)
        self.wavelength = 905  # (nm)
        self.r_min = 0.9  # (m)
        self.r_min_scale = 10
        self.r_max = 120  # (m)
        self.beam_divergence = 0.003  # (rad)
        self.beam_divergence_scale = 10000
        self.min_diameter = 0.05  # (mm)
        self.min_diameter_scale = 100
        self.range_accuracy = 0.09  # (m)
        self.range_accuracy_scale = 100
        self.snowfall_rate = '0.5'
        self.terminal_velocity = '0.2'

        self.rain_rate_title = QLabel('rain rate')
        self.rain_rate_title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.rain_rate_title, self.current_row, 0)

        self.rain_rate_label = QLabel(f'Rr = {self.rain_rate}mm/hr')
        self.rain_rate_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.rain_rate_label, self.current_row, 2)

        self.rain_rate_slider = QSlider(Qt.Horizontal)
        self.rain_rate_slider.setMinimum(1)
        self.rain_rate_slider.setMaximum(200)
        self.rain_rate_slider.setValue(self.rain_rate)
        self.rain_rate_slider.setTickPosition(QSlider.TicksBelow)
        self.rain_rate_slider.setTickInterval(50)
        self.rain_rate_slider.setEnabled(False)

        self.layout.addWidget(self.rain_rate_slider, self.current_row, 1)
        self.rain_rate_slider.valueChanged.connect(self.snowfall_change)

        self.lisa_alpha_label = QLabel('')
        self.lisa_alpha_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.lisa_alpha_label, self.current_row, 3, 1, 2)

        self.current_row += 1

        self.wavelength_title = QLabel('wavelength')
        self.wavelength_title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.wavelength_title, self.current_row, 0)

        self.wavelength_label = QLabel(f'\u03BB = {self.wavelength}nm')
        self.wavelength_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.wavelength_label, self.current_row, 2)

        self.wavelength_slider = QSlider(Qt.Horizontal)
        self.wavelength_slider.setMinimum(905)
        self.wavelength_slider.setMaximum(1550)
        self.wavelength_slider.setValue(self.wavelength)
        self.wavelength_slider.setTickPosition(QSlider.TicksBelow)
        self.wavelength_slider.setTickInterval(15)
        self.wavelength_slider.setEnabled(False)

        self.layout.addWidget(self.wavelength_slider, self.current_row, 1)
        self.wavelength_slider.valueChanged.connect(self.snowfall_change)

        self.cb_lisa = QComboBox()
        self.cb_lisa.addItems(['gunn', 'sekhon', 'rain',
                               'chu_hogg_fog', 'strong_advection_fog', 'moderate_advection_fog',
                               'coast_haze', 'continental_haze', 'moderate_spray', 'strong_spray', 'goodin et al.'])
        self.cb_lisa.currentIndexChanged.connect(self.snowfall_change)
        self.cb_lisa.setEnabled(False)
        self.layout.addWidget(self.cb_lisa, self.current_row, 3, 1, 2)

        self.current_row += 1

        self.r_min_title = QLabel('r_min')
        self.r_min_title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.r_min_title, self.current_row, 0)

        self.r_min_label = QLabel(f'r_min = {self.r_min}m')
        self.r_min_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.r_min_label, self.current_row, 2)

        self.r_min_slider = QSlider(Qt.Horizontal)
        self.r_min_slider.setMinimum(1 * self.r_min_scale)
        self.r_min_slider.setMaximum(10 * self.r_min_scale)
        self.r_min_slider.setValue(int(self.r_min * self.r_min_scale))
        self.r_min_slider.setTickPosition(QSlider.TicksBelow)
        self.r_min_slider.setTickInterval(5)
        self.r_min_slider.setEnabled(False)

        self.layout.addWidget(self.r_min_slider, self.current_row, 1)
        self.r_min_slider.valueChanged.connect(self.snowfall_change)

        self.toggle_seed_btn = QPushButton('seed fixed' if self.fixed_seed else 'seed not fixed')
        self.toggle_seed_btn.setEnabled(False)
        self.toggle_seed_btn.clicked.connect(self.toggle_seed)
        self.layout.addWidget(self.toggle_seed_btn, self.current_row, 3)

        self.toggle_lisa_btn = QPushButton("apply LISA")
        self.toggle_lisa_btn.setEnabled(False)
        self.toggle_lisa_btn.clicked.connect(self.toggle_lisa)
        self.layout.addWidget(self.toggle_lisa_btn, self.current_row, 4)

        self.current_row += 1

        self.r_max_title = QLabel('r_max')
        self.r_max_title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.r_max_title, self.current_row, 0)

        self.r_max_label = QLabel(f'r_max = {self.r_max}m')
        self.r_max_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.r_max_label, self.current_row, 2)

        self.r_max_slider = QSlider(Qt.Horizontal)
        self.r_max_slider.setMinimum(80)
        self.r_max_slider.setMaximum(200)
        self.r_max_slider.setValue(self.r_max)
        self.r_max_slider.setTickPosition(QSlider.TicksBelow)
        self.r_max_slider.setTickInterval(10)
        self.r_max_slider.setEnabled(False)

        self.layout.addWidget(self.r_max_slider, self.current_row, 1)
        self.r_max_slider.valueChanged.connect(self.snowfall_change)

        self.num_unchanged_label = QLabel('')
        self.num_unchanged_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.num_unchanged_label, self.current_row, 3, 1, 2)

        self.current_row += 1

        self.beam_divergence_title = QLabel('beam divergence')
        self.beam_divergence_title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.beam_divergence_title, self.current_row, 0)

        self.beam_divergence_label = QLabel(f'\u2221 = {round(self.beam_divergence * 180 / np.pi, 2)}°')
        self.beam_divergence_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.beam_divergence_label, self.current_row, 2)

        self.beam_divergence_slider = QSlider(Qt.Horizontal)
        self.beam_divergence_slider.setMinimum(int(0.01 / 180 * np.pi * self.beam_divergence_scale))
        self.beam_divergence_slider.setMaximum(int(1 / 180 * np.pi * self.beam_divergence_scale))
        self.beam_divergence_slider.setValue(int(self.beam_divergence * self.beam_divergence_scale))
        self.beam_divergence_slider.setTickPosition(QSlider.TicksBelow)
        self.beam_divergence_slider.setTickInterval(10)
        self.beam_divergence_slider.setEnabled(False)

        self.layout.addWidget(self.beam_divergence_slider, self.current_row, 1)
        self.beam_divergence_slider.valueChanged.connect(self.snowfall_change)

        self.num_removed_label = QLabel('')
        self.num_removed_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.num_removed_label, self.current_row, 3, 1, 2)

        self.current_row += 1

        self.min_diameter_title = QLabel('min_diameter')
        self.min_diameter_title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.min_diameter_title, self.current_row, 0)

        self.min_diameter_label = QLabel(f'\u2300 = {self.min_diameter}mm')
        self.min_diameter_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.min_diameter_label, self.current_row, 2)

        self.min_diameter_slider = QSlider(Qt.Horizontal)
        self.min_diameter_slider.setMinimum(int(0.01 * self.min_diameter_scale))
        self.min_diameter_slider.setMaximum(int(5 * self.min_diameter_scale))
        self.min_diameter_slider.setValue(int(self.min_diameter * self.min_diameter_scale))
        self.min_diameter_slider.setTickPosition(QSlider.TicksBelow)
        self.min_diameter_slider.setTickInterval(100)
        self.min_diameter_slider.setEnabled(False)

        self.layout.addWidget(self.min_diameter_slider, self.current_row, 1)
        self.min_diameter_slider.valueChanged.connect(self.snowfall_change)

        self.num_scattered_label = QLabel('')
        self.num_scattered_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.num_scattered_label, self.current_row, 3, 1, 2)

        self.current_row += 1

        self.range_accuracy_title = QLabel('range accuracy')
        self.range_accuracy_title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.range_accuracy_title, self.current_row, 0)

        self.range_accuracy_label = QLabel(f'~ = {int(self.range_accuracy * 100)}cm')
        self.range_accuracy_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.range_accuracy_label, self.current_row, 2)

        self.range_accuracy_slider = QSlider(Qt.Horizontal)
        self.range_accuracy_slider.setMinimum(int(0.01 * self.range_accuracy_scale))
        self.range_accuracy_slider.setMaximum(int(0.1 * self.range_accuracy_scale))
        self.range_accuracy_slider.setValue(int(self.range_accuracy * self.range_accuracy_scale))
        self.range_accuracy_slider.setTickPosition(QSlider.TicksBelow)
        self.range_accuracy_slider.setTickInterval(1)
        self.range_accuracy_slider.setEnabled(False)

        self.layout.addWidget(self.range_accuracy_slider, self.current_row, 1)
        self.range_accuracy_slider.valueChanged.connect(self.snowfall_change)

        self.num_attenuated_label = QLabel('')
        self.num_attenuated_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.num_attenuated_label, self.current_row, 3, 1, 2)

        self.current_row += 1

        self.layout.addWidget(QHLine(), self.current_row, 0, 1, self.num_columns)

        self.current_row += 1

        self.water_height = 0.0008
        self.water_height_scale = 10000
        self.pavement_height = 0.001
        self.pavement_height_scale = 100000
        self.noise_floor = 0.7
        self.noise_floor_scale = 100
        self.power_factor = 15
        self.flat_earth = False
        self.estimation = 'linear'

        self.water_height_title = QLabel('water height')
        self.water_height_title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.water_height_title, self.current_row, 0)

        self.water_height_label = QLabel(f'h_w = {round(self.water_height * 1000, 1)}mm')
        self.water_height_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.water_height_label, self.current_row, 2)

        self.water_height_slider = QSlider(Qt.Horizontal)
        self.water_height_slider.setMinimum(0)
        self.water_height_slider.setMaximum(int(0.01 * self.water_height_scale))
        self.water_height_slider.setValue(int(self.water_height * self.water_height_scale))
        self.water_height_slider.setTickPosition(QSlider.TicksBelow)
        self.water_height_slider.setTickInterval(10)
        self.water_height_slider.setEnabled(False)

        self.layout.addWidget(self.water_height_slider, self.current_row, 1)
        self.water_height_slider.valueChanged.connect(self.snowfall_change)

        self.estimation_title = QLabel('estimation')
        self.estimation_title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.estimation_title, self.current_row, 4)

        self.current_row += 1

        self.pavement_height_title = QLabel('pavement height')
        self.pavement_height_title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.pavement_height_title, self.current_row, 0)

        self.pavement_height_label = QLabel(f'h_p = {round(self.pavement_height * 1000, 2)}mm')
        self.pavement_height_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.pavement_height_label, self.current_row, 2)

        self.pavement_height_slider = QSlider(Qt.Horizontal)
        self.pavement_height_slider.setMinimum(int(0.0001 * self.pavement_height_scale))
        self.pavement_height_slider.setMaximum(int(0.01 * self.pavement_height_scale))
        self.pavement_height_slider.setValue(round(self.pavement_height * self.pavement_height_scale))
        self.pavement_height_slider.setTickPosition(QSlider.TicksBelow)
        self.pavement_height_slider.setTickInterval(100)
        self.pavement_height_slider.setEnabled(False)

        self.layout.addWidget(self.pavement_height_slider, self.current_row, 1)
        self.pavement_height_slider.valueChanged.connect(self.snowfall_change)

        self.cb_estimation = QComboBox()
        self.cb_estimation.addItems(['linear', 'poly'])
        self.cb_estimation.currentIndexChanged.connect(self.snowfall_change)
        self.cb_estimation.setEnabled(False)
        self.layout.addWidget(self.cb_estimation, self.current_row, 4)

        self.current_row += 1

        self.noise_floor_title = QLabel('noise floor')
        self.noise_floor_title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.noise_floor_title, self.current_row, 0)

        self.noise_floor_label = QLabel(f'n_f = {int(self.noise_floor * 100)}%')
        self.noise_floor_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.noise_floor_label, self.current_row, 2)

        self.noise_floor_slider = QSlider(Qt.Horizontal)
        self.noise_floor_slider.setMinimum(0)
        self.noise_floor_slider.setMaximum(100)
        self.noise_floor_slider.setValue(round(self.noise_floor * self.noise_floor_scale))
        self.noise_floor_slider.setTickPosition(QSlider.TicksBelow)
        self.noise_floor_slider.setTickInterval(10)
        self.noise_floor_slider.setEnabled(False)

        self.layout.addWidget(self.noise_floor_slider, self.current_row, 1)
        self.noise_floor_slider.valueChanged.connect(self.snowfall_change)

        self.flat_earth_btn = QPushButton('flat earch' if self.flat_earth else 'no flat earth')
        self.flat_earth_btn.setEnabled(False)
        self.flat_earth_btn.clicked.connect(self.toggle_flat_earth)
        self.layout.addWidget(self.flat_earth_btn, self.current_row, 3)

        self.toggle_wet_btn = QPushButton("apply wet")
        self.toggle_wet_btn.setEnabled(False)
        self.toggle_wet_btn.clicked.connect(self.toggle_wet)
        self.layout.addWidget(self.toggle_wet_btn, self.current_row, 4)

        self.current_row += 1

        self.power_factor_title = QLabel('power factor')
        self.power_factor_title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.power_factor_title, self.current_row, 0)

        self.power_factor_label = QLabel(f'p_f = {self.power_factor}m')
        self.power_factor_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.power_factor_label, self.current_row, 2)

        self.power_factor_slider = QSlider(Qt.Horizontal)
        self.power_factor_slider.setMinimum(10)
        self.power_factor_slider.setMaximum(20)
        self.power_factor_slider.setValue(self.power_factor)
        self.power_factor_slider.setTickPosition(QSlider.TicksBelow)
        self.power_factor_slider.setTickInterval(1)
        self.power_factor_slider.setEnabled(False)

        self.layout.addWidget(self.power_factor_slider, self.current_row, 1)
        self.power_factor_slider.valueChanged.connect(self.snowfall_change)

        self.current_row += 1

        self.snowfall_change()

    def closeEvent(self, event: QCloseEvent) -> None:

        close_all_windows()

    def add_temporal_cloud(self, index: int):

        assert index in [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4]

        filename = self.file_name
        folder = filename.split('/')[-2]

        filename = filename.replace('SeeingThroughFog', f'SeeingThroughFog/temporal_data')
        filename = filename.replace(folder, f'lidar_{self.sensor}_{self.signal}_history_{index}')
        filename = filename.replace('vlp32', 'vlp32c')

        pc = np.fromfile(filename, dtype=self.d_type)
        pc = pc.reshape((-1, self.num_features))

        pc[:, 3] = np.round(pc[:, 3] * self.intensity_multiplier)

        # filter camera FOV
        if self.show_fov_only:
            calib = get_calib(self.sensor)

            pts_rect = calib.lidar_to_rect(pc[:, 0:3])
            fov_flag = get_fov_flag(pts_rect, (1024, 1920), calib)

            pc = pc[fov_flag]

        min_dist_mask = np.linalg.norm(pc[:, 0:3], axis=1) > MIN_DIST
        pc = pc[min_dist_mask, :]

        self.temporal_pcs.append(copy.deepcopy(pc))

        max_dist_mask = np.linalg.norm(pc[:, 0:3], axis=1) < self.max_distance
        pc = pc[max_dist_mask, :]

        min_height_mask = pc[:, 2] > (self.min_height / 100)  # in m
        pc = pc[min_height_mask, :]

        colors = self.get_colors(pc)

        mesh = gl.GLScatterPlotItem(pos=np.asarray(pc[:, 0:3]), size=self.point_size, color=colors,
                                    glOptions='translucent')
        self.temporal_meshes.append(mesh)

    def update_available_temporal_clouds(self, filename: str):

        if self.dataset == 'DENSE':

            # frame = Path(filename).stem
            # sensor = 'vlp32c' if self.sensor == 'vlp32' else 'hdl64'
            # signal = self.signal
            #
            # future = self.temporal_info[frame][sensor][signal]['future']
            # past = self.temporal_info[frame][sensor][signal]['past']

            past = [0, 0, 0, 0, 0, 0]
            future = [0, 0, 0, 0]

            self.minus1.setEnabled(bool(past[0]))
            self.minus2.setEnabled(bool(past[1]))
            self.minus3.setEnabled(bool(past[2]))
            self.minus4.setEnabled(bool(past[3]))
            self.minus5.setEnabled(bool(past[4]))
            self.minus6.setEnabled(bool(past[5]))

            self.plus1.setEnabled(bool(future[0]))
            self.plus2.setEnabled(bool(future[1]))
            self.plus3.setEnabled(bool(future[2]))
            self.plus4.setEnabled(bool(future[3]))

            # uncheck unavailable temporal frames

            if not past[0]:
                self.minus1.setChecked(False)

            if not past[1]:
                self.minus2.setChecked(False)

            if not past[2]:
                self.minus3.setChecked(False)

            if not past[3]:
                self.minus4.setChecked(False)

            if not past[4]:
                self.minus5.setChecked(False)

            if not past[5]:
                self.minus6.setChecked(False)

            if not future[0]:
                self.plus1.setChecked(False)

            if not future[1]:
                self.plus2.setChecked(False)

            if not future[2]:
                self.plus3.setChecked(False)

            if not future[3]:
                self.plus4.setChecked(False)

        else:

            self.minus1.setEnabled(False)
            self.minus2.setEnabled(False)
            self.minus3.setEnabled(False)
            self.minus4.setEnabled(False)
            self.minus5.setEnabled(False)
            self.minus6.setEnabled(False)

            self.minus1.setChecked(False)
            self.minus2.setChecked(False)
            self.minus3.setChecked(False)
            self.minus4.setChecked(False)
            self.minus5.setChecked(False)
            self.minus6.setChecked(False)

            self.plus1.setEnabled(False)
            self.plus2.setEnabled(False)
            self.plus3.setEnabled(False)
            self.plus4.setEnabled(False)

            self.plus1.setChecked(False)
            self.plus2.setChecked(False)
            self.plus3.setChecked(False)
            self.plus4.setChecked(False)

    def update_temporal_clouds(self) -> None:

        if self.current_mesh:
            try:
                self.viewer.removeItem(self.current_mesh)
            except ValueError:
                pass

        if self.zero.isChecked():
            self.viewer.addItem(self.current_mesh)

        for mesh in self.temporal_meshes:
            try:
                self.viewer.removeItem(mesh)
            except ValueError:
                pass

        self.temporal_pcs = []
        self.temporal_meshes = []

        if self.show_temp:

            if self.minus6.isChecked():
                self.add_temporal_cloud(-6)

            if self.minus5.isChecked():
                self.add_temporal_cloud(-5)

            if self.minus4.isChecked():
                self.add_temporal_cloud(-4)

            if self.minus3.isChecked():
                self.add_temporal_cloud(-3)

            if self.minus2.isChecked():
                self.add_temporal_cloud(-2)

            if self.minus1.isChecked():
                self.add_temporal_cloud(-1)

            if self.plus1.isChecked():
                self.add_temporal_cloud(1)

            if self.plus2.isChecked():
                self.add_temporal_cloud(2)

            if self.plus3.isChecked():
                self.add_temporal_cloud(3)

            if self.plus4.isChecked():
                self.add_temporal_cloud(4)

            for mesh in self.temporal_meshes:
                self.viewer.addItem(mesh)

    def load_results(self) -> None:

        exp_dir = EXPERIMENTS_ROOT / self.experiment_path_box.text()

        test_folders = [x[0] for x in os.walk(exp_dir) if 'epoch' in x[0] and 'test' in x[0]]

        self.result_dict = {}

        for test_folder in test_folders:
            key = test_folder.split('/')[-1]

            pkl_path = Path(test_folder) / 'result.pkl'

            with open(pkl_path, 'rb') as f:
                list_to_be_sorted = pkl.load(f)

            self.result_dict[key] = sorted(list_to_be_sorted, key=lambda d: d['frame_id'])

        if self.result_dict is not None:
            self.visualize_predictions_btn.setEnabled(True)
            self.prediction_threshold_slider.setEnabled(True)

    def visualize_predictions(self) -> None:

        split = self.cb_splits.currentText()

        if self.cb_sensors.currentText() == 'vlp32':
            split = f'{split}_vlp32'

        if 'test' in split:
            split = split.replace('_dror', '')

            pred_dict = self.result_dict[split][self.index]

            assert self.file_name.split('/')[-1].split('.')[0] == pred_dict['frame_id'], f'frame missmatch ' \
                                                                                         f"{self.file_name.split('/')[-1].split('.')[0]} != {pred_dict['frame_id']}"

            lookup = {'Car': 0,
                      'Pedestrian': 1,
                      'Cyclist': 2}

            predictions = np.zeros((pred_dict['boxes_lidar'].shape[0], 9))
            predictions[:, 0:-2] = pred_dict['boxes_lidar']
            predictions[:, 7] = np.array([lookup[name] for name in pred_dict['name']])
            predictions[:, 8] = pred_dict['score']

            self.predictions = predictions

            self.add_predictions()

    def add_predictions(self) -> None:

        self.show_predictions = True
        self.visualize_predictions_btn.setText('hide predictions')
        self.visualize_predictions_btn.setEnabled(True)
        self.prediction_threshold_slider.setEnabled(True)

        for prediction in self.predictions:

            x, y, z, w, l, h, rotation, category, score = prediction

            dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)

            rotation = np.rad2deg(rotation) + 90

            try:
                color = DET_COLORS[int(category)]
            except IndexError:
                if CAMERA_READY:
                    color = BLACK
                else:
                    color = WHITE

            try:
                box = gl.GLBoxItem(QtGui.QVector3D(1, 1, 1), color=color, line_width=self.line_width)
            except TypeError:
                box = gl.GLBoxItem(QtGui.QVector3D(1, 1, 1), color=color)

            box.setSize(l, w, h)

            box.translate(-l / 2, -w / 2, -h / 2)
            box.rotate(angle=rotation, x=0, y=0, z=1)
            box.translate(x, y, z)

            self.prediction_boxes.append((dist, score, box))

        for dist, score, box in self.prediction_boxes:
            if score * 100 > self.prediction_threshold:
                self.viewer.addItem(box)


    def populate_dense_splits(self) -> List[str]:

        split_folder = Path(__file__).parent.absolute() / 'lib' / 'LiDAR_fog_sim' / 'SeeingThroughFog' / 'splits'

        splits = []

        for file in os.listdir(split_folder):

            if file.endswith('.txt') and 'FOVstrongest3000' in file \
                    and '0.35' not in file \
                    and '_day' not in file \
                    and 'vlp32' not in file \
                    and '_night' not in file \
                    and 'test_wet' not in file \
                    and 'FOVlast3000' not in file:
                string = file.replace('.txt', '')
                string = string.replace('_alpha_0.45', '')
                string = string.replace('_FOVstrongest3000', '')

                splits.append(string)

                self.dense_split_paths.append(split_folder / file)

        self.dense_split_paths = sorted(self.dense_split_paths)

        return sorted(splits)

    def split_selection_change(self) -> None:

        self.reset_dense_features()

        self.file_list = []

        sensor_flag = 'vlp32' if self.sensor == 'vlp32' else ''

        split = self.cb_splits.currentText()

        if 'dror' in split:
            split = split.replace('dror', f'{sensor_flag}_FOV{self.signal}3000_dror_alpha_0.45')
        else:
            split = f'{split}_{sensor_flag}_FOV{self.signal}3000'

        split = split.replace('__', '_')

        # open file and read the content in a list
        with open(f'lib/LiDAR_fog_sim/SeeingThroughFog/splits/{split}.txt', 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                file_path = Path(DENSE) / f'{line[:-1].replace(",", "_")}.bin'

                # add item to the list
                self.file_list.append(str(file_path))

        self.file_list = sorted(self.file_list)

        self.index = 0
        self.set_dense()
        self.show_pointcloud(self.file_list[self.index])

    def sensor_selection_change(self) -> None:

        # reset live inference
        self.config = None
        self.model = None

        self.sensor = self.cb_sensors.currentText()
        self.show_pointcloud(self.file_list[self.index], force_reload=True)

    def signal_selection_change(self) -> None:

        # reset live inference
        self.config = None
        self.model = None

        self.signal = self.cb_signals.currentText()
        self.show_pointcloud(self.file_list[self.index], force_reload=True)

    def update_labels(self) -> None:

        self.p.alpha = self.alpha_slider.value() / self.p.alpha_scale
        self.alpha_label.setText(f"\u03B1 = {self.p.alpha}")

        self.p.mor = np.log(20) / self.p.alpha
        self.mor_label.setText(f'meteorological optical range (MOR) = {round(self.p.mor, 2)}m')

        self.p.beta_scale = 1000 * self.p.mor
        self.p.beta = self.beta_slider.value() / self.p.beta_scale
        self.beta_label.setText(f"\u03B2 = {round(self.p.beta * self.p.mor, 3)} / MOR")

        self.p.gamma = self.gamma_slider.value() / self.p.gamma_scale
        self.gamma_label.setText(f"\u0393 = {self.p.gamma}")
        self.p.beta_0 = self.p.gamma / np.pi

        if self.file_list:
            self.show_pointcloud(self.file_list[self.index])

    def reset_dense_features(self) -> None:

        self.boxes = {}

        self.alpha_slider.setEnabled(False)
        self.beta_slider.setEnabled(False)
        self.gamma_slider.setEnabled(False)

        self.simulated_fog = False
        self.simulated_fog_dense = False
        self.toggle_simulated_fog_btn.setText('add our fog')
        self.toggle_simulated_fog_dense_btn.setText('add STF fog')

        self.toggle_temp_btn.setEnabled(False)

        self.minus1.setEnabled(False)
        self.minus2.setEnabled(False)
        self.minus3.setEnabled(False)
        self.minus4.setEnabled(False)
        self.minus5.setEnabled(False)
        self.minus6.setEnabled(False)

        self.plus1.setEnabled(False)
        self.plus2.setEnabled(False)
        self.plus3.setEnabled(False)
        self.plus4.setEnabled(False)

    def reset_custom_values(self) -> None:

        self.sensor = 'hdl64'
        self.signal = 'strongest'
        self.toggle_img_btn.setEnabled(False)
        self.show_fov_only_btn.setEnabled(False)

        self.min_value = 0
        self.max_value = 63
        self.num_features = 5
        self.dataset = None
        self.success = False
        self.d_type = np.float32
        self.intensity_multiplier = 1
        self.color_name = self.color_dict[self.color_feature]

    def dror_change(self) -> None:

        self.dror_alpha = self.dror_alpha_slider.value() / self.dror_alpha_scale
        self.dror_alpha_label.setText(f"\u03B1 = {self.dror_alpha}")

        self.dror_beta = self.dror_beta_slider.value()
        self.dror_beta_label.setText(f"\u03B2 = {self.dror_beta}")

        self.dror_k_min = self.dror_k_min_slider.value()
        self.dror_k_min_label.setText(f"k_min = {self.dror_k_min}")

        self.dror_sr_min = self.dror_sr_min_slider.value()
        self.dror_sr_min_label.setText(f"sr_min = {self.dror_sr_min}cm")

        if self.file_list:
            self.show_pointcloud(self.file_list[self.index], force_reload=True)

    def snowfall_change(self) -> None:

        ########
        # LISA #
        ########

        self.mode = self.cb_lisa.currentText()

        self.rain_rate = self.rain_rate_slider.value()
        self.rain_rate_label.setText(f"Rr = {self.rain_rate}mm/hr")

        self.wavelength = self.wavelength_slider.value()
        self.wavelength_label.setText(f"\u03BB = {self.wavelength}nm")

        self.r_min = self.r_min_slider.value() / self.r_min_scale
        self.r_min_label.setText(f'r_min = {self.r_min}m')

        self.r_max = self.r_max_slider.value()
        self.r_max_label.setText(f'r_max = {self.r_max}m')

        self.beam_divergence = self.beam_divergence_slider.value() / self.beam_divergence_scale
        self.beam_divergence_label.setText(f'\u2221 = {round(self.beam_divergence * 180 / np.pi, 2)}°')

        self.min_diameter = self.min_diameter_slider.value() / self.min_diameter_scale
        self.min_diameter_label.setText(f'\u2300 = {self.min_diameter}mm')

        self.range_accuracy = self.range_accuracy_slider.value() / self.range_accuracy_scale
        self.range_accuracy_label.setText(f'~ = {int(self.range_accuracy * 100)}cm')

        self.lisa = LISA(wavelength=self.wavelength, mode=self.mode, r_min=self.r_min, r_max=self.r_max,
                         beam_divergence=self.beam_divergence, min_diameter=self.min_diameter,
                         range_accuracy=self.range_accuracy, signal=self.signal, show_progressbar=True)

        ########
        # ours #
        ########

        self.water_height = self.water_height_slider.value() / self.water_height_scale
        self.water_height_label.setText(f'h_w = {round(self.water_height * 1000, 1)}mm')

        self.pavement_height = self.pavement_height_slider.value() / self.pavement_height_scale
        self.pavement_height_label.setText(f'h_p = {round(self.pavement_height * 1000, 2)}mm')

        self.noise_floor = self.noise_floor_slider.value() / self.noise_floor_scale
        self.noise_floor_label.setText(f'n_f = {int(self.noise_floor * 100)}%')

        self.power_factor = self.power_factor_slider.value()
        self.power_factor_label.setText(f'p_f = {self.power_factor}')

        self.estimation = self.cb_estimation.currentText()
        self.snowfall_rate = self.cb_snowfall_rate.currentText()
        self.terminal_velocity = self.cb_velocity.currentText()

        if self.file_list:
            self.show_pointcloud(self.file_list[self.index], force_reload=True)

    def prediction_threshold_slider_change(self) -> None:

        self.prediction_threshold = self.prediction_threshold_slider.value()
        self.prediction_threshold_label.setText(str(self.prediction_threshold))

        if self.file_list:
            self.show_pointcloud(self.file_list[self.index])

    def max_distance_slider_change(self) -> None:

        self.max_distance = self.max_distance_slider.value()
        self.max_distance_label.setText(f'{self.max_distance} m')

        if self.file_list:
            self.show_pointcloud(self.file_list[self.index])

    def min_height_slider_change(self) -> None:

        self.min_height = self.min_height_slider.value()
        self.min_height_label.setText(f'{self.min_height} cm')

        if self.file_list:
            self.show_pointcloud(self.file_list[self.index])

    def color_slider_change(self) -> None:

        self.color_feature = self.color_slider.value()

        self.color_name = self.color_dict[self.color_feature]
        self.color_label.setText(self.color_name)

        if self.current_mesh:

            if Path(self.file_name).suffix == '.pickle':
                self.show_pcdet_dict(self.file_name)
            else:
                self.show_pointcloud(self.file_name, needs_reload=False)

    def check_index_overflow(self) -> None:

        if self.index == -1:
            self.index = len(self.file_list) - 1

        if self.index >= len(self.file_list):
            self.index = 0

    def decrement_index(self) -> None:

        self.predictions = None
        self.prediction_boxes = []

        if self.index != -1:

            self.index -= 1
            self.check_index_overflow()

            if Path(self.file_list[self.index]).suffix == ".pickle":
                self.show_pcdet_dict(self.file_list[self.index])
            else:
                self.show_pointcloud(self.file_list[self.index])

    def increment_index(self) -> None:

        self.predictions = None
        self.prediction_boxes = []

        if SAVE_IMAGES and ROTATE:
            self.viewer.opts['azimuth'] = (self.viewer.opts['azimuth'] + ROTATE) % 360   # rotate camera each frame

        if self.index != -1:

            self.index += 1
            self.check_index_overflow()

            if Path(self.file_list[self.index]).suffix == ".pickle":
                self.show_pcdet_dict(self.file_list[self.index])
            else:
                self.show_pointcloud(self.file_list[self.index])

    def set_pc_det(self, before: bool) -> None:

        self.min_value = 0
        self.max_value = 63
        self.extension = 'pickle'
        self.d_type = np.float32
        self.intensity_multiplier = 1

        if before:

            self.dataset = 'before'
            self.num_features = 5
            self.color_dict[6] = 'channel'

        else:

            self.dataset = 'after'
            self.num_features = 4
            self.color_dict[6] = 'not available'

    def set_kitti(self) -> None:
        self.dataset = 'KITTI'
        self.min_value = -1
        self.max_value = -1
        self.num_features = 4
        self.extension = 'bin'
        self.d_type = np.float32
        self.show_fov_only = False
        self.intensity_multiplier = 255
        self.color_dict[6] = 'not available'

    def load_kitti(self) -> None:

        self.reset_dense_features()
        self.image_window.hide()
        self.viewer.setCameraPosition(pos=QVector3D(10, 0, 0), distance=20, elevation=15, azimuth=180)

        self.file_list = []

        # open file and read the content in a list
        with open('lib/LiDAR_fog_sim/file_lists/KITTI.txt', 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                file_path = Path(KITTI) / line[:-1]

                # add item to the list
                self.file_list.append(str(file_path))

        self.index = 0
        self.set_kitti()
        self.show_pointcloud(self.file_list[self.index])

    def set_audi(self) -> None:

        self.dataset = 'A2D2'
        self.min_value = 0
        self.max_value = 4
        self.num_features = 5
        self.extension = 'npz'
        self.d_type = np.float32
        self.show_fov_only = False
        self.intensity_multiplier = 1
        self.color_dict[6] = 'lidar_id'

    def load_audi(self) -> None:

        self.reset_dense_features()
        self.image_window.hide()

        self.viewer.setCameraPosition(pos=QVector3D(45, 0, 0), distance=95, elevation=15, azimuth=180)

        self.file_list = []

        # open file and read the content in a list
        with open('lib/LiDAR_fog_sim/file_lists/A2D2.txt', 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                file_path = Path(AUDI) / line[:-1]

                # add item to the list
                self.file_list.append(str(file_path))

        self.index = 0
        self.set_audi()
        self.show_pointcloud(self.file_list[self.index])

    def set_honda(self) -> None:

        self.dataset = 'Honda3D'
        self.min_value = 0
        self.max_value = 63
        self.num_features = 5
        self.extension = 'ply'
        self.d_type = np.float32
        self.show_fov_only = False
        self.intensity_multiplier = 1
        self.color_dict[6] = 'channel'

    def load_honda(self) -> None:

        self.reset_dense_features()
        self.image_window.hide()

        self.viewer.setCameraPosition(pos=QVector3D(10, 0, 0), distance=20, elevation=15, azimuth=180)

        self.file_list = []

        # open file and read the content in a list
        with open('lib/LiDAR_fog_sim/file_lists/Honda3D.txt', 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                file_path = Path(HONDA) / line[:-1]

                # add item to the list
                self.file_list.append(str(file_path))

        self.index = 0
        self.set_honda()
        self.show_pointcloud(self.file_list[self.index])

    def set_argo(self) -> None:

        self.dataset = 'Argoverse'
        self.min_value = 0
        self.max_value = 31
        self.num_features = 5
        self.extension = 'ply'
        self.d_type = np.float32
        self.show_fov_only = False
        self.intensity_multiplier = 1
        self.color_dict[6] = 'channel'

    def load_argo(self) -> None:

        self.reset_dense_features()
        self.image_window.hide()

        self.viewer.setCameraPosition(pos=QVector3D(10, 0, 0), distance=85, elevation=20, azimuth=180)

        self.file_list = []

        # open file and read the content in a list
        with open('lib/LiDAR_fog_sim/file_lists/Argoverse.txt', 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                file_path = Path(ARGO) / line[:-1]

                # add item to the list
                self.file_list.append(str(file_path))

        self.index = 0
        self.set_argo()
        self.show_pointcloud(self.file_list[self.index])

    def set_dense(self) -> None:

        self.dataset = 'DENSE'
        self.min_value = 0
        self.max_value = 63
        self.num_features = 5
        self.extension = 'bin'
        self.d_type = np.float32
        self.intensity_multiplier = 1
        self.color_dict[6] = 'channel'

        self.toggle_img_btn.setEnabled(True)
        self.show_fov_only_btn.setEnabled(True)

        self.toggle_temp_btn.setEnabled(True)

        self.sensor = self.cb_sensors.currentText()
        self.signal = self.cb_signals.currentText()

        self.cb_splits.setEnabled(True)
        self.cb_sensors.setEnabled(True)
        self.cb_signals.setEnabled(True)

    def load_dense(self) -> None:

        self.reset_dense_features()

        if self.show_img:
            self.image_window.show()
        else:
            self.image_window.hide()

        self.viewer.setCameraPosition(pos=QVector3D(15, 0, 0), distance=22.55, elevation=30, azimuth=180)

        self.file_list = []

        # open file and read the content in a list
        with open('lib/LiDAR_fog_sim/file_lists/DENSE.txt', 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                file_path = Path(DENSE) / line[:-1]

                # add item to the list
                self.file_list.append(str(file_path))

        scene = ''
        # scene = '2018-02-12_14-32-26_00450' # teaser scene
        index = [idx for idx, s in enumerate(self.file_list) if scene in s][0]

        self.index = index if scene else 0
        self.set_dense()
        self.cb_splits.setCurrentText('all')
        self.show_pointcloud(self.file_list[self.index])

    def set_nuscenes(self) -> None:

        self.dataset = 'nuScenes'
        self.min_value = 0
        self.max_value = 31
        self.num_features = 5
        self.extension = 'bin'
        self.intensity_multiplier = 1
        self.color_dict[6] = 'channel'

    def load_nuscenes(self) -> None:

        self.viewer.setCameraPosition(pos=QVector3D(2, 5, 0), distance=60, elevation=25, azimuth=270)

        self.reset_dense_features()
        self.image_window.hide()

        self.file_list = []

        with open('lib/LiDAR_fog_sim/file_lists/nuScenes.pkl', 'rb') as f:
            file_list = pkl.load(f)

        for file in file_list:
            self.file_list.append(str(NUSCENES) + file)

        self.index = 0
        self.set_nuscenes()
        self.show_pointcloud(self.file_list[self.index])

    def set_lyft(self) -> None:

        self.dataset = 'LyftL5'
        self.min_value = 0
        self.max_value = 16
        self.num_features = 5
        self.extension = 'bin'
        self.intensity_multiplier = 1
        self.color_dict[6] = 'channel'

    def load_lyft(self) -> None:

        self.reset_dense_features()
        self.image_window.hide()

        self.viewer.setCameraPosition(pos=QVector3D(0, 0, 0), distance=80, elevation=20, azimuth=0)

        self.file_list = []

        # open file and read the content in a list
        with open('lib/LiDAR_fog_sim/file_lists/LyftL5.txt', 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                file_path = Path(LYFT) / line[:-1]

                # add item to the list
                self.file_list.append(str(file_path))

        self.index = 0
        self.set_lyft()
        self.show_pointcloud(self.file_list[self.index])

    def set_waymo(self) -> None:

        self.dataset = 'WaymoOpenDataset'
        self.min_value = -1
        self.max_value = -1
        self.num_features = 4
        self.extension = 'bin'
        self.d_type = np.float32
        self.show_fov_only = False
        self.intensity_multiplier = 255
        self.color_dict[6] = 'not available'

    def load_waymo(self) -> None:

        self.reset_dense_features()
        self.image_window.hide()

        self.viewer.setCameraPosition(pos=QVector3D(5, 0, 0), distance=45, elevation=25, azimuth=180)

        self.file_list = []

        # open file and read the content in a list
        with open('lib/LiDAR_fog_sim/file_lists/WAYMO.txt', 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                file_path = Path(WAYMO) / line[:-1]

                # add item to the list
                self.file_list.append(str(file_path))

        self.index = 0
        self.set_waymo()
        self.show_pointcloud(self.file_list[self.index])

    def set_panda(self) -> None:

        self.dataset = 'PandaSet'
        self.min_value = 0
        self.max_value = 1
        self.num_features = 5
        self.extension = 'pkl.gz'
        self.d_type = np.float32
        self.show_fov_only = False
        self.intensity_multiplier = 1
        self.color_dict[6] = 'lidar_id'

    def load_panda(self) -> None:

        self.reset_dense_features()
        self.image_window.hide()

        self.viewer.setCameraPosition(pos=QVector3D(20, 10, 0), distance=95, elevation=20, azimuth=222)

        self.file_list = []

        # open file and read the content in a list
        with open('lib/LiDAR_fog_sim/file_lists/PandaSet.txt', 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                file_path = Path(PANDA) / line[:-1]

                # add item to the list
                self.file_list.append(str(file_path))

        self.index = 0
        self.set_panda()
        self.show_pointcloud(self.file_list[self.index])

    def set_apollo(self) -> None:

        self.dataset = 'Apollo'
        self.min_value = -1
        self.max_value = -1
        self.num_features = 4
        self.extension = 'bin'
        self.d_type = np.float32
        self.show_fov_only = False
        self.intensity_multiplier = 255
        self.color_dict[6] = 'not available'

    def load_apollo(self) -> None:

        self.reset_dense_features()
        self.image_window.hide()

        self.viewer.setCameraPosition(pos=QVector3D(10, 0, 0), distance=20, elevation=15, azimuth=180)

        self.file_list = []

        # open file and read the content in a list
        with open('lib/LiDAR_fog_sim/file_lists/Apollo.txt', 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                file_path = Path(APOLLO) / line[:-1]

                # add item to the list
                self.file_list.append(str(file_path))

        self.index = 0
        self.set_apollo()
        self.show_pointcloud(self.file_list[self.index])

    def show_directory_dialog(self) -> None:

        self.reset_dense_features()

        # directory = Path(os.getenv("HOME")) / 'Downloads'
        directory = '/scratch/mhahner/LiDARSnowDemoVideos/2018-02-05_12-06-04/lidar_hdl64_s3/velodyne_pointclouds/strongest_echo_egomotion'

        if self.lastDir:
            directory = self.lastDir

        dir_name = QFileDialog.getExistingDirectory(self, "Open Directory", str(directory),
                                                    QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)

        if dir_name:
            self.lastDir = Path(dir_name)
            self.create_file_list(dir_name)

    def get_index(self, filename: str) -> int:

        try:
            return self.file_list.index(str(filename))

        except ValueError:
            logging.warning(f'{filename} not found in self.file_list')
            return -1

    def toggle_fov(self) -> None:

        self.show_fov_only = not self.show_fov_only
        self.show_fov_only_btn.setText('show full 360°' if self.show_fov_only else 'show camera FOV only')
        self.show_pointcloud(self.file_list[self.index], force_reload=True)

    def toggle_snow(self) -> None:

        self.apply_snow = not self.apply_snow
        self.toggle_snow_btn.setText('remove snow' if self.apply_snow else 'apply snow')
        self.show_pointcloud(self.file_list[self.index], force_reload=True)

    def toggle_wet(self) -> None:

        self.apply_wet = not self.apply_wet
        self.toggle_wet_btn.setText('remove wet' if self.apply_wet else 'apply wet')
        self.show_pointcloud(self.file_list[self.index], force_reload=True)

    def toggle_flat_earth(self) -> None:

        self.flat_earth = not self.flat_earth
        self.flat_earth_btn.setText('flat earth' if self.flat_earth else 'no flat earth')
        self.snowfall_change()

    def toggle_seed(self) -> None:

        self.fixed_seed = not self.fixed_seed
        self.toggle_seed_btn.setText('seed fixed' if self.fixed_seed else 'seed not fixed')
        self.snowfall_change()

    def toggle_lisa(self) -> None:

        self.apply_lisa = not self.apply_lisa
        self.toggle_lisa_btn.setText('remove LISA' if self.apply_lisa else 'apply lisa')
        self.show_pointcloud(self.file_list[self.index], force_reload=True)


    def toggle_dror_cube(self) -> None:

        self.show_dror_cube = not self.show_dror_cube

        if self.show_dror_cube:
            self.toggle_cube_btn.setText('hide cube')
            self.viewer.addItem(self.dror_cube)
        else:
            self.toggle_cube_btn.setText('show cube')
            self.viewer.removeItem(self.dror_cube)

    def toggle_temp_visibility(self) -> None:

        self.show_temp = not self.show_temp
        self.toggle_temp_btn.setText('hide temporal' if self.show_temp else 'show temporal')
        self.update_temporal_clouds()

    def toggle_image_visibility(self) -> None:

        self.show_img = not self.show_img

        if self.show_img:

            self.image_window.show()
            self.toggle_img_btn.setText('hide image')
            self.populate_image(self.file_list[self.index])

        else:

            self.image_window.hide()
            self.toggle_img_btn.setText('show image')

    def toggle_simulated_fog(self) -> None:

        if self.file_list is not None and 'extraction' not in self.file_name:

            self.simulated_fog = not self.simulated_fog

            if self.simulated_fog:

                self.toggle_simulated_fog_btn.setText('remove our fog')
                self.alpha_slider.setEnabled(True)
                self.beta_slider.setEnabled(True)
                self.gamma_slider.setEnabled(True)

            else:

                self.toggle_simulated_fog_btn.setText('add our fog')
                self.alpha_slider.setEnabled(False)
                self.beta_slider.setEnabled(False)
                self.gamma_slider.setEnabled(False)

            self.show_pointcloud(self.file_list[self.index], force_reload=True)

    def get_dror_mask(self, filename: str, pc: np.ndarray) -> np.ndarray:

        before = len(pc)

        if live_DROR_available:

            pc_copy = copy.deepcopy(pc)

            mask = dynamic_radius_outlier_filter(pc,
                                                 alpha=self.dror_alpha,
                                                 beta=self.dror_beta,
                                                 k_min=self.dror_k_min,
                                                 sr_min=self.dror_sr_min / 100)  # convert to meters

            cube_mask = get_cube_mask(pc_copy)
            pc_copy = pc_copy[cube_mask]

            cube_indices = dynamic_radius_outlier_filter(pc_copy,
                                                         alpha=self.dror_alpha,
                                                         beta=self.dror_beta,
                                                         k_min=self.dror_k_min,
                                                         sr_min=self.dror_sr_min / 100)  # convert to meters

            self.update_dror_title(cube_indices=cube_indices)

        else:  # try loading precomputed dict

            mask = dynamic_radius_outlier_filter(sensor=self.sensor, signal=self.signal, before=before,
                                                 variant='full', filename=filename)

            self.update_dror_title(filename=filename)

        after = (mask == True).sum()

        diff = before - after

        if diff > 0:
            percent = 100 - int(after / (before / 100))
            text = f'{self.dror_intensity}: {diff:,} (~{percent}%) out of {before:,} points filtered'.replace(',', '.')
            self.dror_headline.setText(text)

        return mask

    def toggle_dror(self) -> None:

        self.apply_dror = not self.apply_dror

        if self.apply_dror:
            self.toggle_dror_btn.setText('revert DROR')
        else:
            self.toggle_dror_btn.setText('apply DROR')
            self.dror_headline.setText(self.dror_intensity)

        self.show_pointcloud(self.file_list[self.index], force_reload=True)

    def toggle_simulated_fog_dense(self) -> None:

        if self.file_list is not None and 'extraction' not in self.file_name:

            self.simulated_fog_dense = not self.simulated_fog_dense

            if self.simulated_fog_dense:

                self.toggle_simulated_fog_dense_btn.setText('remove STF fog')
                self.alpha_slider.setEnabled(True)
                self.beta_slider.setEnabled(True)
                self.gamma_slider.setEnabled(True)

            else:

                self.toggle_simulated_fog_dense_btn.setText('add STF fog')
                self.alpha_slider.setEnabled(False)
                self.beta_slider.setEnabled(False)
                self.gamma_slider.setEnabled(False)

            self.show_pointcloud(self.file_list[self.index], force_reload=True)

    def toggle_predictions(self) -> None:

        self.show_predictions = not self.show_predictions

        if self.show_predictions:
            self.visualize_predictions_btn.setText('hide predictions')
        else:
            self.visualize_predictions_btn.setText('show predictions')

        if self.file_list:
            self.show_pointcloud(self.file_list[self.index])

    def create_file_list(self, dirname: str, filename: str = None, extension: str = None) -> None:

        if extension:
            file_list = [y for x in os.walk(dirname) for y in glob(os.path.join(x[0], f'*.{extension}'))]
        else:
            file_list = [y for x in os.walk(dirname) for y in glob(os.path.join(x[0], f'*.{self.extension}'))]

        self.file_list = sorted(file_list)

        # with open('lib/LiDAR_fog_sim/file_lists/Argoverse.txt', 'w') as filehandle:
        #     filehandle.writelines(f'{Path(file).parent.parent.parent.parent.name}/'
        #                           f'{Path(file).parent.parent.parent.name}/'
        #                           f'{Path(file).parent.parent.name}/'
        #                           f'{Path(file).parent.name}/'
        #                           f'{Path(file).name}\n' for file in self.file_list)

        if len(self.file_list) > 0:

            if filename is None:
                filename = self.file_list[0]

            self.index = self.get_index(filename)

            if Path(self.file_list[self.index]).suffix == ".pickle":
                self.show_pcdet_dict(self.file_list[self.index])
            else:
                self.show_pointcloud(self.file_list[self.index])

    def reset_viewer(self) -> None:

        self.num_info.setText(f'sequence_size: {len(self.file_list)}')

        self.min_fog_response = np.inf
        self.max_fog_response = 0
        self.num_fog_responses = 0

        self.viewer.items = []

        if self.show_dror_cube:
            self.viewer.addItem(self.dror_cube)

    def run_inference(self) -> None:

        if self.current_experiment != self.experiment_path_box.text():
            self.config = None
            self.model = None

        if self.config is None:
            self.init_config()

        if self.logger is None:
            self.init_logger()

        self.init_data()

        if self.model is None:
            self.init_model()

        if self.zero.isChecked():
            list_of_pcs = [self.current_pc] + self.temporal_pcs
        else:
            list_of_pcs = self.temporal_pcs

        try:
            points = np.vstack(list_of_pcs)
        except ValueError as ve:
            self.logger.error(f'{ve} - There are no points to process.')
            return None

        for i, batch_dict in enumerate(self.eval_loader):
            input_dict = {'points': points,
                          'frame_id': batch_dict['frame_id']}

            data_dict = self.eval_set.prepare_data(data_dict=input_dict)
            data_dict = self.eval_set.collate_batch([data_dict])
            load_data_to_gpu(data_dict)

            with torch.no_grad():
                pred_dicts, ret_dict = self.model.forward(data_dict)

        pred_dict = pred_dicts[0]

        predictions = np.zeros((pred_dict['pred_boxes'].shape[0], 9))
        predictions[:, 0:-2] = pred_dict['pred_boxes'].cpu()
        predictions[:, 7] = pred_dict['pred_labels'].cpu()
        predictions[:, 8] = pred_dict['pred_scores'].cpu()

        self.predictions = predictions

        self.add_predictions()

    def init_config(self) -> None:

        exp_dir = EXPERIMENTS_ROOT / self.experiment_path_box.text()
        self.current_experiment = self.experiment_path_box.text()

        cfg_path = None

        for file in os.listdir(exp_dir):

            if file.endswith(".yaml"):
                cfg_path = os.path.join(exp_dir, file)

        config = cfg_from_yaml_file(cfg_path, cfg)

        sensor_flag = '_vlp32' if self.sensor == 'vlp32' else ''

        config['OPTIMIZATION']['BATCH_SIZE_PER_GPU'] = 1
        config['DATA_CONFIG']['INFO_PATH']['test'] = [f'pkl/dense_infos_all{sensor_flag}_FOV{self.signal}3000.pkl']

        self.config = config

    def init_logger(self) -> None:

        log_file = Path(tempfile.gettempdir()) / 'inference.out'
        self.logger = common_utils.create_logger(log_file, rank=self.config.LOCAL_RANK, log_level=logging.DEBUG)

    def init_data(self) -> None:

        sample_str = self.file_name_label.text().replace('.bin', '')

        # -----------------------create dataloader & network & optimizer---------------------------
        self.eval_set, self.eval_loader, self.sampler = build_dataloader(
            dataset_cfg=self.config.DATA_CONFIG,
            class_names=self.config.CLASS_NAMES,
            batch_size=1,
            dist=False, workers=0, logger=self.logger, training=False,
            root_path=DENSE.parent,
            search_str=sample_str
        )

    def init_model(self) -> None:

        exp_dir = EXPERIMENTS_ROOT / self.experiment_path_box.text()

        ckpt_path = None

        for file in os.listdir(exp_dir):
            if file.endswith(".pth"):
                ckpt_path = os.path.join(exp_dir, file)

        model = build_network(model_cfg=self.config.MODEL,
                              num_class=len(self.config.CLASS_NAMES),
                              dataset=self.eval_set)

        model.cuda()

        model.load_params_from_file(filename=ckpt_path, to_cpu=False, logger=self.logger)

        model.eval()  # before wrap to DistributedDataParallel to support fixed some parameters

        self.model = model

    def show_pcdet_dict(self, filename: str) -> None:

        self.reset_viewer()
        self.simulated_fog = False
        self.simulated_fog_dense = False

        if self.simulated_fog:

            self.toggle_simulated_fog_btn.setText('remove our fog')
            self.alpha_slider.setEnabled(True)
            self.beta_slider.setEnabled(True)
            self.gamma_slider.setEnabled(True)

        else:

            self.toggle_simulated_fog_btn.setText('add our fog')
            self.alpha_slider.setEnabled(False)
            self.beta_slider.setEnabled(False)
            self.gamma_slider.setEnabled(False)

        self.file_name = filename

        self.set_pc_det('before' in filename)

        self.cb_splits.setEnabled(False)
        self.cb_sensors.setEnabled(False)
        self.cb_signals.setEnabled(False)
        self.toggle_simulated_fog_btn.setEnabled(True)
        self.toggle_simulated_fog_dense_btn.setEnabled(True)

        pcdet_dict = pkl.load(open(filename, "rb"))

        ##########
        # points #
        ##########

        pc = pcdet_dict['points']

        self.log_string(pc)

        colors = self.get_colors(pc)

        mesh = gl.GLScatterPlotItem(pos=np.asarray(pc[:, 0:3]), size=self.point_size, color=colors,
                                    glOptions='translucent')
        self.current_mesh = mesh
        self.current_pc = copy.deepcopy(pc)

        self.viewer.addItem(mesh)

        #########
        # boxes #
        #########

        self.boxes = {}

        self.create_boxes(pcdet_dict['gt_boxes'])

    def create_boxes(self, annotations):

        # create annotation boxes
        for annotation in annotations:

            x, y, z, w, l, h, rotation, category = annotation

            rotation = np.rad2deg(rotation) + 90

            try:
                color = COLORS[int(category) - 1]
            except IndexError:
                color = (255, 255, 255, 255)

            try:
                box = gl.GLBoxItem(QtGui.QVector3D(1, 1, 1), color=color, line_width=self.line_width)
            except TypeError:
                box = gl.GLBoxItem(QtGui.QVector3D(1, 1, 1), color=color)

            box.setSize(l, w, h)
            box.translate(-l / 2, -w / 2, -h / 2)
            box.rotate(angle=rotation, x=0, y=0, z=1)
            box.translate(x, y, z)

            self.viewer.addItem(box)

            #################
            # heading lines #
            #################

            p1 = [-l / 2, -w / 2, -h / 2]
            p2 = [l / 2, -w / 2, h / 2]

            pts = np.array([p1, p2])

            l1 = gl.GLLinePlotItem(pos=pts, width=2 / 3, color=color, antialias=True, mode='lines')
            l1.rotate(angle=rotation, x=0, y=0, z=1)
            l1.translate(x, y, z)

            self.viewer.addItem(l1)

            p3 = [-l / 2, -w / 2, h / 2]
            p4 = [l / 2, -w / 2, -h / 2]

            pts = np.array([p3, p4])

            l2 = gl.GLLinePlotItem(pos=pts, width=2 / 3, color=color, antialias=True, mode='lines')
            l2.rotate(angle=rotation, x=0, y=0, z=1)
            l2.translate(x, y, z)

            self.viewer.addItem(l2)

            distance = np.linalg.norm([x, y, z], axis=0)
            self.boxes[distance] = (box, l1, l2)

    def update_dror_title(self, cube_indices: np.ndarray = None, filename: str = '') -> None:

        if self.dataset == 'DENSE':

            num_snow = -1

            if cube_indices is None:

                cube_path = f'{DROR}/alpha_0.45/all/{self.sensor}/{self.signal}/crop'
                name = Path(filename).name.replace('.bin', '')
                cube_file = f'{cube_path}/{name}.pkl'

                if Path(cube_file).exists():
                    with open(cube_file, 'rb') as f:
                        cube_indices = pkl.load(f)

                    num_snow = len(cube_indices)

                # because we can not create a range with stop=np.inf
                intensity = f'heavy [> {int(DROR_LEVELS["light"][1])}]'

                for key, value in DROR_LEVELS.items():

                    if num_snow == -1:
                        intensity = 'file not found'
                        break

                    if num_snow in range(value[0], value[1] + 1):
                        intensity = f'{key} [{int(value[0])} - {int(value[1])}]'
                        break

                self.dror_intensity = f'DROR ({num_snow} => {intensity})'

            else:

                self.dror_intensity = f'DROR'

            self.dror_headline.setText(self.dror_intensity)

    def show_pointcloud(self, filename: str, force_reload: bool = False, needs_reload: bool = True) -> None:

        if force_reload:
            self.current_pc = None

        # pprint(self.viewer.opts)

        self.reset_viewer()
        self.update_available_temporal_clouds(filename)

        self.cb_splits.setEnabled(False)
        self.cb_sensors.setEnabled(False)
        self.cb_signals.setEnabled(False)

        if self.dataset == 'DENSE':
            self.cb_velocity.setEnabled(True)
            self.cb_estimation.setEnabled(True)
            self.flat_earth_btn.setEnabled(True)
            self.toggle_wet_btn.setEnabled(True)
            self.toggle_snow_btn.setEnabled(True)
            self.cb_snowfall_rate.setEnabled(True)
            self.noise_floor_slider.setEnabled(True)
            self.power_factor_slider.setEnabled(True)
            self.water_height_slider.setEnabled(True)
            self.pavement_height_slider.setEnabled(True)
        else:
            self.cb_velocity.setEnabled(False)
            self.cb_estimation.setEnabled(False)
            self.flat_earth_btn.setEnabled(False)
            self.toggle_wet_btn.setEnabled(False)
            self.toggle_snow_btn.setEnabled(False)
            self.cb_snowfall_rate.setEnabled(False)
            self.noise_floor_slider.setEnabled(False)
            self.power_factor_slider.setEnabled(False)
            self.water_height_slider.setEnabled(False)
            self.pavement_height_slider.setEnabled(False)

        self.cb_lisa.setEnabled(True)
        self.r_min_slider.setEnabled(True)
        self.r_max_slider.setEnabled(True)
        self.toggle_seed_btn.setEnabled(True)
        self.toggle_lisa_btn.setEnabled(True)
        self.rain_rate_slider.setEnabled(True)
        self.wavelength_slider.setEnabled(True)
        self.min_diameter_slider.setEnabled(True)
        self.range_accuracy_slider.setEnabled(True)
        self.beam_divergence_slider.setEnabled(True)
        self.toggle_simulated_fog_btn.setEnabled(True)
        self.toggle_simulated_fog_dense_btn.setEnabled(True)

        if self.dataset == 'DENSE' and self.show_fov_only and CUDA_AVAILABLE:
            self.inference_btn.setEnabled(True)
        else:
            self.inference_btn.setEnabled(False)

        if self.dataset == 'DENSE' or live_DROR_available:

            self.update_dror_title(filename=filename)

            self.toggle_dror_btn.setEnabled(True)
            self.toggle_cube_btn.setEnabled(True)

            if live_DROR_available:
                self.dror_alpha_slider.setEnabled(True)
                self.dror_beta_slider.setEnabled(True)
                self.dror_k_min_slider.setEnabled(True)
                self.dror_sr_min_slider.setEnabled(True)

        else:

            self.dror_intensity = 'DROR'
            self.dror_headline.setText(self.dror_intensity)

            self.toggle_cube_btn.setEnabled(False)
            self.toggle_dror_btn.setEnabled(False)
            self.dror_alpha_slider.setEnabled(False)
            self.dror_beta_slider.setEnabled(False)
            self.dror_k_min_slider.setEnabled(False)
            self.dror_sr_min_slider.setEnabled(False)

        if self.file_name == filename and self.current_pc is not None:

            # reuse the current pointcloud if the filename stays the same
            pc = self.current_pc

        else:

            if self.tmp_value:  # it means, snow has been applied before
                self.num_unchanged_label.setText('')
                self.num_removed_label.setText('')
                self.num_scattered_label.setText('')
                self.num_attenuated_label.setText('')
                self.max_value = self.tmp_value
                self.tmp_value = None

            if not self.result_dict:  # it means, that only live inference is available
                self.predictions = None
                self.prediction_boxes = []
                self.prediction_threshold_slider.setEnabled(False)
                self.visualize_predictions_btn.setEnabled(False)
                self.visualize_predictions_btn.setText('show predictions')

            self.file_name = filename
            pc = self.load_pointcloud(filename)

        if self.apply_dror:
            mask = self.get_dror_mask(filename, pc)
            pc = pc[mask, :]

        self.success = False

        # filter camera FOV
        if self.show_fov_only:
            calib = get_calib(self.sensor)

            pts_rect = calib.lidar_to_rect(pc[:, 0:3])
            fov_flag = get_fov_flag(pts_rect, (1024, 1920), calib)

            pc = pc[fov_flag]

        min_dist_mask = np.linalg.norm(pc[:, 0:3], axis=1) > MIN_DIST
        pc = pc[min_dist_mask, :]

        max_dist_mask = np.linalg.norm(pc[:, 0:3], axis=1) < self.max_distance
        pc = pc[max_dist_mask, :]

        min_height_mask = pc[:, 2] > (self.min_height / 100)  # in m
        pc = pc[min_height_mask, :]

        if 'extraction' in filename:
            intensity_mask = pc[:, 3] <= self.threshold
            pc = pc[intensity_mask, :]

        colors = self.get_colors(pc)

        mesh = gl.GLScatterPlotItem(pos=np.asarray(pc[:, 0:3]), size=self.point_size, color=colors,
                                    glOptions='translucent')
        self.current_mesh = mesh

        if self.success and self.zero.isChecked():
            self.viewer.addItem(mesh)
            self.zero.setEnabled(True)

        curve = self.lisa.Nd(self.lisa.D, self.rain_rate)  # density of the particles (m^-3)
        a = self.lisa.alpha(curve)
        self.lisa_alpha_label.setText(f'\u27F9 \u03B1 = {a}')

        rain_rate = snowfall_rate_to_rainfall_rate(float(self.snowfall_rate), float(self.terminal_velocity))
        self.rainfall_rate_title.setText(f'\u27F9 rainfall rate = {round(rain_rate, 2)} mm/h')
        already_augmented = False
        occupancy = compute_occupancy(float(self.snowfall_rate), float(self.terminal_velocity))
        snowflake_file_prefix = f'{self.mode}_{rain_rate}_{occupancy}'

        if self.apply_wet and self.apply_snow and needs_reload:
            self.reset_viewer()

            stats, pc = augment(pc=pc, only_camera_fov=self.show_fov_only,
                                particle_file_prefix=snowflake_file_prefix, noise_floor=self.noise_floor,
                                beam_divergence=float(np.degrees(self.beam_divergence)),
                                shuffle=True, show_progressbar=True)

            num_attenuated, num_removed, avg_intensity_diff = stats

            pc = ground_water_augmentation(pc,
                                           water_height=self.water_height,
                                           pavement_depth=self.pavement_height,
                                           noise_floor=self.noise_floor,
                                           power_factor=self.power_factor,
                                           flat_earth=self.flat_earth,
                                           estimation_method=self.estimation,
                                           debug=False, delta=self.dror_alpha, replace=False)

            num_unchanged = (pc[:, 4] == 0).sum()
            num_attenuated = (pc[:, 4] == 1).sum()
            num_scattered = (pc[:, 4] == 2).sum()

            self.num_unchanged_label.setText(f'num_unchanged: {num_unchanged}')
            self.num_removed_label.setText(f'num_removed: {num_removed}')
            self.num_scattered_label.setText(f'num_scattered: {num_scattered}')
            self.num_attenuated_label.setText(f'num_attenuated: {num_attenuated} (\u00B5 = -{avg_intensity_diff})')

            self.tmp_value = self.max_value
            self.max_value = 3
            colors = self.get_colors(pc)
            mesh = gl.GLScatterPlotItem(pos=np.asarray(pc[:, 0:3]), size=self.point_size, color=colors,
                                        glOptions='translucent')
            self.current_mesh = mesh
            self.viewer.addItem(mesh)

            already_augmented = True

        if self.apply_wet and not already_augmented and needs_reload:
            self.reset_viewer()

            pc = ground_water_augmentation(pc,
                                           water_height=self.water_height,
                                           pavement_depth=self.pavement_height,
                                           noise_floor=self.noise_floor,
                                           power_factor=self.power_factor,
                                           flat_earth=self.flat_earth,
                                           estimation_method=self.estimation,
                                           debug=False, delta=self.dror_alpha, replace=True)

            self.tmp_value = self.max_value
            self.max_value = 3
            colors = self.get_colors(pc)
            mesh = gl.GLScatterPlotItem(pos=np.asarray(pc[:, 0:3]), size=self.point_size, color=colors,
                                        glOptions='translucent')
            self.current_mesh = mesh
            self.viewer.addItem(mesh)

        if self.apply_snow and not already_augmented and needs_reload:
            self.reset_viewer()

            stats, pc = augment(pc=pc, only_camera_fov=self.show_fov_only,
                                particle_file_prefix=snowflake_file_prefix, noise_floor=self.noise_floor,
                                beam_divergence=float(np.degrees(self.beam_divergence)),
                                shuffle=True, show_progressbar=True)

            num_attenuated, num_removed, avg_intensity_diff = stats

            num_unchanged = (pc[:, 4] == 0).sum()
            num_scattered = (pc[:, 4] == 2).sum()

            self.num_unchanged_label.setText(f'num_unchanged: {num_unchanged}')
            self.num_removed_label.setText(f'num_removed: {num_removed}')
            self.num_scattered_label.setText(f'num_scattered: {num_scattered}')
            self.num_attenuated_label.setText(f'num_attenuated: {num_attenuated} (\u00B5 = -{avg_intensity_diff})')

            self.tmp_value = self.max_value
            self.max_value = 3
            colors = self.get_colors(pc)
            mesh = gl.GLScatterPlotItem(pos=np.asarray(pc[:, 0:3]), size=self.point_size, color=colors,
                                        glOptions='translucent')
            self.current_mesh = mesh
            self.viewer.addItem(mesh)

        if self.apply_lisa and needs_reload:

            self.reset_viewer()

            before_lisa = np.zeros((pc.shape[0], 4))
            before_lisa[:, :3] = copy.deepcopy(pc[:, :3])
            before_lisa[:, 3] = copy.deepcopy(pc[:, 3] / 255)

            self.lisa = LISA(wavelength=self.wavelength, mode=self.mode, r_min=self.r_min, r_max=self.r_max,
                             beam_divergence=self.beam_divergence, min_diameter=self.min_diameter,
                             range_accuracy=self.range_accuracy, signal=self.signal, show_progressbar=True)

            try:
                after_lisa = self.lisa.augment(pc=before_lisa, Rr=self.rain_rate, fixed_seed=self.fixed_seed)
            except TypeError:
                after_lisa = self.lisa.augment(pc=before_lisa)

            num_removed = (after_lisa[:, 4] == 0).sum()
            num_scattered = (after_lisa[:, 4] == 2).sum()
            num_attenuated = (after_lisa[:, 4] == 1).sum()

            intensity_diff_sum = after_lisa[:, 5].sum()

            if num_attenuated > 0:
                avg_intensity_diff = int(np.round(intensity_diff_sum * 255 / num_attenuated, 0))
            else:
                avg_intensity_diff = 0

            self.num_unchanged_label.setText('')
            self.num_removed_label.setText(f'num_removed: {num_removed}')
            self.num_scattered_label.setText(f'num_scattered: {num_scattered}')
            self.num_attenuated_label.setText(f'num_attenuated: {num_attenuated} (\u00B5 = -{avg_intensity_diff})')

            after_lisa[:, 3] = np.round(after_lisa[:, 3] * 255)

            if pc.shape[1] < 5:
                pc = np.zeros((pc.shape[0], pc.shape[1] + 1))

            pc[:, :5] = after_lisa[:, :5]

            # remove points that where moved to origin
            pc = pc[np.where(pc[:, 4] != 0)]

            self.tmp_value = self.max_value
            self.max_value = 3
            colors = self.get_colors(pc)
            mesh = gl.GLScatterPlotItem(pos=np.asarray(pc[:, 0:3]), size=self.point_size, color=colors,
                                        glOptions='translucent')
            self.current_mesh = mesh
            self.viewer.addItem(mesh)

        if self.simulated_fog and self.success:
            self.toggle_simulated_fog_dense_btn.setEnabled(False)

            self.reset_viewer()

            pc, simulated_fog_pc, info_dict = simulate_fog(self.p, pc, self.noise, self.gain,
                                                           self.noise_variant)

            self.simulated_fog_pc = simulated_fog_pc

            self.min_fog_response = info_dict['min_fog_response']
            self.max_fog_response = info_dict['max_fog_response']
            self.num_fog_responses = info_dict['num_fog_responses']

            colors = self.get_colors(pc)
            mesh = gl.GLScatterPlotItem(pos=np.asarray(pc[:, 0:3]), size=self.point_size, color=colors,
                                        glOptions='translucent')
            self.viewer.addItem(mesh)

        if self.simulated_fog_dense and self.success:
            self.toggle_simulated_fog_btn.setEnabled(False)

            self.reset_viewer()

            B = BetaRadomization(beta=float(self.p.alpha), seed=0)
            B.propagate_in_time(10)

            arguments = Namespace(sensor_type='Velodyne HDL-64E S3D', fraction_random=0.05)
            n_features = pc.shape[1]

            pc = haze_point_cloud(pc, B, arguments)
            pc = pc[:, :n_features]

            colors = self.get_colors(pc)
            mesh = gl.GLScatterPlotItem(pos=np.asarray(pc[:, 0:3]), size=self.point_size, color=colors,
                                        glOptions='translucent')
            self.viewer.addItem(mesh)

        self.current_pc = copy.deepcopy(pc)
        self.log_string(pc=pc)

        if self.dataset == 'DENSE':

            self.cb_splits.setEnabled(True)
            self.cb_sensors.setEnabled(True)
            self.cb_signals.setEnabled(True)
            self.populate_dense_boxes(filename)

        else:

            self.cb_splits.setCurrentText('all')
            self.cb_splits.setEnabled(False)
            self.cb_sensors.setEnabled(False)
            self.cb_signals.setEnabled(False)

        if self.show_img:
            self.populate_image(filename)

        self.update_temporal_clouds()

        if self.boxes:

            try:  # if heading lines are available
                for box_distance, (box, l1, l2) in self.boxes.items():
                    if box_distance < self.p.r_range:
                        self.viewer.addItem(box)
                        self.viewer.addItem(l1)
                        self.viewer.addItem(l2)
            except TypeError:
                for box_distance, box in self.boxes.items():
                    if box_distance < self.p.r_range:
                        self.viewer.addItem(box)

        if self.result_dict and self.show_predictions:
            self.visualize_predictions()

        if self.prediction_boxes and self.show_predictions:
            self.add_predictions()

        if SAVE_IMAGES:

            pixmap = QPixmap(self.size())
            self.render(pixmap)
            cropped = pixmap.toImage().copy(20, 400, 1000, 430)                     # 21:9 aspect ratio

            date = str(self.lastDir).split('/')[-4]
            name = Path(filename).stem + '.png'

            weather = 'clear'

            w_h = self.water_height_slider.value() * 10
            p_h = self.pavement_height_slider.value()

            water_rate = int(w_h / p_h * 100)
            rain_rate = str(int(snowfall_rate_to_rainfall_rate(float(self.snowfall_rate),
                                                               float(self.terminal_velocity))))

            if self.apply_lisa:
                weather = f'lisa_{rain_rate}'

            if self.apply_snow:
                weather = f'snow_{rain_rate}'

            if self.apply_wet:
                weather = f'wet_{water_rate}'

            if self.apply_snow and self.apply_wet:
                weather = f'snow+wet_{rain_rate}_{water_rate}'

            save_path = Path.home() / 'Downloads' / date / weather / self.color_name
            save_path.mkdir(parents=True, exist_ok=True)

            cropped.save(f'{save_path}/{name}', "PNG", -1)

    def populate_image(self, filename) -> None:

        if self.dataset == 'DENSE':
            img_file_name = Path(filename).name.replace('.bin', '.png')

            image_path = Path(filename).parent.parent / 'cam_stereo_left_lut' / img_file_name

            width, height = self.image_window.image_label.width(), self.image_window.image_label.height()

            pixmap = QPixmap(str(image_path))
            pixmap = pixmap.scaled(width, height, Qt.KeepAspectRatio)

            self.image_window.image_label.setPixmap(pixmap)

    def populate_dense_boxes(self, filename):

        root = str(Path(__file__).parent.absolute() / 'lib/LiDAR_fog_sim/SeeingThroughFog/tools/DatasetViewer/calibs')
        tf_tree = 'calib_tf_tree_full.json'

        name_camera_calib = 'calib_cam_stereo_left.json'

        sensor_names = {'hdl64': 'lidar_hdl64_s3_roof',
                        'vlp32': 'lidar_vlp32_roof'}

        rgb_calib = load_calib_data(root, name_camera_calib, tf_tree, velodyne_name=sensor_names[self.sensor])

        camera_to_velodyne_rgb = rgb_calib[1]

        label_path = Path(filename).parent.parent / 'gt_labels' / 'cam_left_labels_TMP'

        recording = Path(filename).stem  # here without '.txt' as it will be added in read_label function
        label_file = os.path.join(label_path, recording)
        self.label = read_label(label_file, label_path, camera_to_velodyne=camera_to_velodyne_rgb)

        self.boxes = {}

        # create annotation boxes
        for annotation in self.label:

            if annotation['identity'] in ['PassengerCar', 'Pedestrian', 'RidableVehicle']:

                x = annotation['posx_lidar']
                y = annotation['posy_lidar']
                z = annotation['posz_lidar']

                if annotation['identity'] == 'PassengerCar':
                    color = COLORS[0]
                elif annotation['identity'] == 'Pedestrian':
                    color = COLORS[1]
                else:
                    color = COLORS[2]

                distance = np.sqrt(x ** 2 + y ** 2 + z ** 2)

                try:
                    box = gl.GLBoxItem(QtGui.QVector3D(1, 1, 1), color=color, line_width=self.line_width)
                except TypeError:
                    box = gl.GLBoxItem(QtGui.QVector3D(1, 1, 1), color=color)

                box.setSize(annotation['length'], annotation['width'], annotation['height'])
                box.translate(-annotation['length'] / 2, -annotation['width'] / 2, -annotation['height'] / 2)
                box.rotate(angle=-annotation['rotz'] * 180 / np.pi, x=0, y=0, z=1)
                box.rotate(angle=-annotation['roty'] * 180 / np.pi, x=0, y=1, z=0)
                box.rotate(angle=-annotation['rotx'] * 180 / np.pi, x=1, y=0, z=0)
                box.translate(0, 0, annotation['height'] / 2)
                box.translate(x, y, z)

                self.boxes[distance] = box

    def log_string(self, pc: np.ndarray) -> None:

        try:

            distance = np.linalg.norm(pc[:, 0:3], axis=1)

            self.p.r_range = max(distance)

            log_string = f'pts ' + f'{len(pc)}'.rjust(6, ' ') + ' | ' + \
                         f'max_dist ' + f'{int(self.p.r_range)}'.rjust(3, ' ') + ' m | ' + \
                         f'i [ ' + f'{int(min(pc[:, 3]))}'.rjust(3, ' ') + \
                         f', ' + f'{int(max(pc[:, 3]))}'.rjust(3, ' ') + ']' + ' ' + \
                         f'median ' + f'{int(np.round(np.median(pc[:, 3])))}'.rjust(3, ' ') + ' ' + \
                         f'\u03BC ' + f'{int(np.round(np.mean(pc[:, 3])))}'.rjust(3, ' ') + ' ' + \
                         f'\u03C3 ' + f'{int(np.round(np.std(pc[:, 3])))}'.rjust(3, ' ')

            if self.num_fog_responses > 0:
                range_fog_response_string = f'fog [ ' + f'{int(self.min_fog_response)}'.rjust(3, ' ') + \
                                            f', ' + f'{int(self.max_fog_response)}'.rjust(3, ' ') + ']'
                num_fog_responses_string = f'soft ' + f'{int(self.num_fog_responses)}'.rjust(6, ' ')
                num_remaining_string = f'hard ' + f'{int(len(self.current_pc) - self.num_fog_responses)}'.rjust(6, ' ')

                log_string = log_string + ' | ' + \
                             range_fog_response_string + ' ' + num_fog_responses_string + ' ' + num_remaining_string

        except ValueError:

            self.p.r_range = 0

            log_string = f'num_pts ' + f'{len(pc)}'.rjust(6, ' ')

        self.log_info.setText(log_string)

    def get_colors(self, pc: np.ndarray) -> np.ndarray:

        # create colormap
        if self.color_feature == 0:

            self.success = True
            feature = pc[:, 0]
            min_value = np.min(feature)
            max_value = np.max(feature)

        elif self.color_feature == 1:

            self.success = True
            feature = pc[:, 1]
            min_value = np.min(feature)
            max_value = np.max(feature)

        elif self.color_feature == 2:

            self.success = True
            feature = pc[:, 2]
            min_value = -1.5
            max_value = 0.5

        elif self.color_feature == 3:

            self.success = True
            feature = pc[:, 3]
            min_value = 0
            max_value = 255

        elif self.color_feature == 4:

            self.success = True
            feature = np.linalg.norm(pc[:, 0:3], axis=1)

            try:
                min_value = np.min(feature)
                max_value = np.max(feature)
            except ValueError:
                min_value = 0
                max_value = np.inf

        elif self.color_feature == 5:

            self.success = True
            feature = np.arctan2(pc[:, 1], pc[:, 0]) + np.pi
            min_value = 0
            max_value = 2 * np.pi

        else:  # self.color_feature == 6:

            try:
                feature = pc[:, 4]
                self.success = True

            except IndexError:
                feature = pc[:, 3]

            min_value = self.min_value
            max_value = self.max_value

        norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)

        if self.color_feature == 5:
            cmap = cm.hsv  # cyclic
        else:
            cmap = cm.jet  # sequential

        m = cm.ScalarMappable(norm=norm, cmap=cmap)

        colors = m.to_rgba(feature)
        colors[:, [2, 1, 0, 3]] = colors[:, [0, 1, 2, 3]]
        colors[:, 3] = 1

        return colors

    def load_pointcloud(self, filename: str) -> np.ndarray:

        self.reset_custom_values()

        if 'KITTI' in filename:
            self.set_kitti()

        if 'DENSE' in filename:
            self.set_dense()

            # override sensor
            filename = filename.replace('hdl64', self.sensor)

            # override signal
            filename = filename.replace('strongest', self.signal)

        if 'nuScenes' in filename:
            self.set_nuscenes()

        if 'Lyft' in filename:
            self.set_lyft()

        if 'Waymo' in filename:
            self.set_waymo()

        if 'Honda' in filename:
            self.set_honda()

        if 'A2D2' in filename:
            self.set_audi()

        if 'PandaSet' in filename:
            self.set_panda()

        if 'Apollo' in filename:
            self.set_apollo()

        if 'Argoverse' in filename:
            self.set_argo()

        self.color_name = self.color_dict[self.color_feature]
        self.color_label.setText(self.color_name)

        if self.extension == 'ply':

            pc = self.load_from_ply(filename)

        elif self.extension == 'npz':

            pc = self.load_from_npz(filename)

        elif 'pkl' in self.extension:

            pc = self.load_from_pkl(filename)

        else:  # assume bin file

            pc = np.fromfile(filename, dtype=self.d_type)
            pc = pc.reshape((-1, self.num_features))

        pc[:, 3] = np.round(pc[:, 3] * self.intensity_multiplier)

        if self.dataset == 'Honda3D':

            self.file_name_label.setText(f'{Path(filename).parent.name}/'
                                         f'{Path(filename).name}')

        elif self.dataset == 'PandaSet' or self.dataset == 'Apollo':

            self.file_name_label.setText(f'{Path(filename).parent.parent.name}/'
                                         f'{Path(filename).parent.name}/'
                                         f'{Path(filename).name}')

        else:

            self.file_name_label.setText(str(Path(filename).name))

        self.show_fov_only_btn.setText('show full 360°' if self.show_fov_only else 'show camera FOV only')

        return pc

    def load_from_pkl(self, filename: str) -> np.ndarray:

        if filename.endswith('gz'):

            with gzip.open(filename, 'rb') as f:
                data = pkl.load(f)

        else:

            with open(filename, 'rb') as f:
                data = pkl.load(f)

        if self.dataset == 'PandaSet':
            pc = data.drop(columns=['t']).values
        else:
            pc = data.values

        return pc

    def load_from_ply(self, filename: str) -> np.ndarray:

        with open(filename, 'rb') as f:
            plydata = PlyData.read(f)

        pc = np.array(plydata.elements[0].data.tolist())[:]

        if self.dataset == 'Honda3D':
            pc = np.delete(pc, [3, 4, 5, 6, 7, 8, 9, 12], 1)
        elif self.dataset == 'Argoverse':
            pc = pc
        else:
            pc = np.delete(pc, [4, 5, 6], 1)

        return pc

    def load_from_npz(self, filename: str) -> np.ndarray:

        npz = np.load(filename)

        pc_dict = {}

        for key in npz.keys():
            pc_dict[key] = npz[key]

        pc = None

        if self.dataset == 'A2D2':
            pc = np.column_stack((pc_dict['points'],
                                  pc_dict['reflectance'],
                                  pc_dict['lidar_id']))

        return pc

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:

        logging.debug("enter")

        mimeData = event.mimeData()
        mimeList = mimeData.formats()
        filename = None

        if "text/uri-list" in mimeList:
            filename = mimeData.data("text/uri-list")
            filename = str(filename, encoding="utf-8")
            filename = filename.replace("file://", "").replace("\r\n", "").replace("%20", " ")
            filename = Path(filename)

        if filename.exists() and (filename.suffix == ".bin" or
                                  filename.suffix == ".ply" or
                                  filename.suffix == ".pickle"):
            event.accept()
            self.droppedFilename = filename
            self.extension = filename.suffix.replace('.', '')
        else:
            event.ignore()
            self.droppedFilename = None

    def dropEvent(self, event: QDropEvent) -> None:

        if self.droppedFilename:
            self.create_file_list(str(Path(self.droppedFilename).parent), self.droppedFilename)


if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    logging.debug(pandas.__version__)

    app = QtGui.QApplication([])
    lidar_window = LidarWindow()
    lidar_window.show()
    app.exec_()
