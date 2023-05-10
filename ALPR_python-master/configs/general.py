import yaml
import torch

main_configs = {
    'max_cosine_distance': 0.5,
    'nn_budget': None,
    'nms_max_overlap': 1.0,
    'model_filename': 'ALPR_python-master/deepsort_models/mars-small128.pb',
    'spot_config': 'spot_config.json',
    'wl_path': 'ALPR_python-master/whiteList.txt',
    'nn_distance_metric': 'cosine',
    'batch_size': 8,
    'ocr_conf_threshold': 97,
    'request_timeout': 10
}

barrier_configs = {
    'barrier_open_url': 'http://192.168.2.214/api/ForceOpen',
    'barrier_close_url': 'http://192.168.2.214/api/ForceClose'
}

draw_configs = {
    'spot_line_color': (255, 0, 84),
    'spot_line_thickness': 2,
    'spot_font_scale': 0.5,
    'spot_font_color': (0, 0, 0),
    'spot_font_thickness': 2,
    'time_org_coordinates': (10, 20),
    'time_font_scale': 0.8,
    'time_font_color': (0, 0, 0),
    'time_font_thickness': 2,
    # 'status_line_color': (0, 0, 255),
    # 'status_line_thickness': 2,
    # 'status_busy_color': (0, 0, 255),
    # 'status_free_color': (255, 0, 0),
    'plate_color': (0, 255, 0),
    'plate_thickness': 2,
    'plate_font_scale': 0.75,
    'plate_font_color': (255, 255, 255),
    'plate_font_thickness': 2
}

camera_configs = {
    # 'ips': ['37.186.127.77'],
    # 'login': 'admin',
    # 'password': '22dvcgwqw7',
    # camera_urls = [
    # 'rtsp://admin:22dvcgwqw7@192.168.1.221:554/cam/realmonitor?channel=1&subtype=0',
    # 'rtsp://admin:22dvcgwqw7@192.168.1.222:554/cam/realmonitor?channel=1&subtype=0',
    # 'rtsp://admin:22dvcgwqw7@192.168.1.223:554/cam/realmonitor?channel=1&subtype=0',
    # 'rtsp://admin:25802580Code@192.168.1.227:554/cam/realmonitor?channel=1&subtype=0',
    # 'rtsp://admin:22dvcgwqw7@192.168.1.225:554/cam/realmonitor?channel=1&subtype=0'
    # ],
    'camera_urls': [
        'rtsp://admin:d12345678@192.168.2.226:554/Streaming/channels/101',
        # 'rtsp://admin:22dvcgwqw7@37.252.72.204:556/cam/realmonitor?channel=1&subtype=0',
        # 'rtsp://admin:22dvcgwqw7@37.252.72.204:557/cam/realmonitor?channel=1&subtype=0',
        # 'rtsp://admin:25802580Code@37.252.72.204:558/cam/realmonitor?channel=1&subtype=0',
        # 'rtsp://admin:22dvcgwqw7@37.252.72.204:559/cam/realmonitor?channel=1&subtype=0'
    ],
    'mac_addresses': [
        '9c:14:63:64:ec:c0',
        '9c:14:63:64:e5:f0',
        '9c:14:63:64:e8:ac',
        '9c:14:63:64:eb:0a',
        '9c:14:63:64:e8:af'
    ],
    'spots_per_camera': [1, 2, 2, 2, 2],
    'locator_url': 'https://gaz.locator.am/api/camera_info'
}

user_configs = yaml.safe_load(open("ALPR_python-master/user_config.yaml", "r"))

general_configs = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    # 'device': torch.device('cpu'),
    '_key': "ThisistheonlyKeyThatwillwork123@%^*&((??__+(I_GAS"
}

nms_configs = {
    'conf_thres': 0.4,
    'iou_thres': 0.5,
    'max_det': 100
}
