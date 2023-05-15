from setuptools import setup
from Cython.Build import cythonize
from getmac import get_mac_address as gma
import os

# data = gma()
# with open(fname, 'a') as f:
#     f.write(f'\nmc_address = {[data]}')

setup(
    ext_modules = cythonize(['augmentations.py', 'yolo_anchors.py', 'craft_processor_kisat.py', 'craft_processors.py', 'yolo_common.py', 'craft_functions.py', 'craft_utils.py',
                            'crft.py', 'craft_augmentations.py', 'yolo_dataloader.py', 'ocr_dataset.py', 'yolo_car_detection_inference.py', 'tracker_detection.py', 'yolo_dataset_download.py', 'ocr_inference.py', 
                            'yolo_experimental.py', 'yolo_exports.py', 'ocr_feature_extractor.py', 'flut.py', 'frmod.py', 'yolo_general_utils.py', 'tracker_detection_generator.py', 'models_decrypt.py', 
                            'craft_normalize.py', 'tracker_kalman_filter.py', 'infr.py', 'tracker_iou.py', 'license_plate_utils.py', 'tracker_metrics.py', 'yolo_metrics.py', 
                            'timer_checkpoints.py', 'VideoCapture.py', 'vslite.py', 'v2.py', 'v1lite.py', 'ocr_model.py', 'tracker_distance.py', 'yolo_plotting_utils.py', 
                            'ocr_attention.py', 'plate_predictor.py', 'tracker_nms.py', 'rfnt.py', 'send_data.py', 'ocr_BLSTM.py', 'plate_detection_ssd.py', 'yolo_torch_utils.py', 
                            'track.py', 'tracker.py', 'ocr_CTC.py', 'utils.py', 'craft_backbone.py', 'yolo_specific_modules.py', 'ocr_TPS.py', 'augmentation_utils.py','draw_spots.py'],
    language_level = '3',gdb_debug=True)
)
