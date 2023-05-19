from collections import OrderedDict
from os import supports_dir_fd
import cv2
import pandas as pd
import torch
import numpy as np
from send_data import post_data
import time
from shapely.geometry import Polygon
from configs.general import user_configs, camera_configs, draw_configs


def csv_save(name, prediction, date_time, conf, ):
    new = pd.DataFrame(
        {'Path': [f'./result/{prediction}_{date_time}_{conf}.jpg'], 'Time': [date_time], 'Plate': [prediction],
         'Confidence': [conf]})
    new.to_csv(name, mode='a', index=False, header=False)


def send_data(car_image, plate_image, cam_id, prediction, date_time, conf, spot_id):
    path = './' + user_configs['image_save_path'] + '/'
    name = ''
    if conf >= 96:
        if 'license_number' in user_configs['image_name_info']:
            name += prediction
        if 'time' in user_configs['image_name_info']:
            name += '_' + date_time
        if 'confidence' in user_configs['image_name_info']:
            name += '_' + str(int(conf * 100))
        if 'camera_id' in user_configs['image_name_info']:
            name += '_' + str(cam_id)

        print('[{}]: Cam ID: {}  Frame ID: {}  Prediction: {}   confidence: {}%'.format(date_time, cam_id, spot_id,
                                                                                        prediction, conf))
        data = {"camera_id": str(cam_id), "spot_id": str(spot_id),
                "mac_address": camera_configs['mac_addresses'][cam_id], "license_number": prediction,
                "confidence": str(conf), "time": date_time}
        cv2.imwrite(path + 'plate_' + name + '.jpg', plate_image)
        cv2.imwrite(path + 'vehicle_' + name + '.jpg', car_image)
        x = post_data(camera_configs['locator_url'], data, name='ALPR')

        return x
    else:
        if 'time' in user_configs['image_name_info']:
            name += '_' + date_time
        if 'camera_id' in user_configs['image_name_info']:
            name += '_' + str(cam_id)
        cv2.imwrite(path + 'unknown_' + name + '.jpg', car_image)
        return False


def load_whiteList(path):
    wl = []
    with open(path, 'r') as f:
        wl = f.read().split('\n')
    return wl


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def show_images(images, width, height, k=3, fps=''):
    j = -1
    if len(images) == 1:
        return cv2.putText(images[0], fps, (500, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    lists = []
    for i, im in enumerate(images):

        if i % k == 0:
            j += 1
            lists.append([])

        lists[j].append(im)
    if len(lists) > 1:
        resize_width, resize_height = width // k, int(np.ceil(height // (len(lists))))
    else:
        resize_width, resize_height = 640, 480
    base_image = np.zeros((resize_height * len(lists), k * resize_width, 3), dtype=np.uint8)
    current_y = 0
    for lst in lists:
        current_x = 0

        for image in lst:
            image = cv2.resize(image, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
            base_image[current_y: image.shape[0] + current_y, current_x: image.shape[1] + current_x, :] = image

            current_x += image.shape[1]
        current_y += image.shape[0]

    return cv2.putText(base_image, fps, (500, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)


def convert_xywh(bbox):
    x1, y1, x2, y2 = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
    x, y, w, h = x1, y1, x2 - x1, y2 - y1
    return np.array([x, y, w, h])


def convert_polygon(bbox):
    # print(bbox)
    x1, y1, x2, y2 = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
    # x, y, w, h = x1, y1, x2 - x1, y2 - y1
    a, b, c, d = (x1, y1), (x1, y2), (x2, y1), (x2, y2)
    return Polygon([a, b, c, d])


def get_centre(box):
    return (box[1][0] - box[0][0]) / 2, (box[1][1] - box[0][1]) / 2


def check_box(spots, all_coordinates, last_ids):
    spot_dict = {}
    car_ind_dict = {}
    for idx, info in all_coordinates.items():
        img, _, boxes = info[0], info[1], info[2]
        for j, b in enumerate(boxes):
            plate_pol = convert_polygon(b)

            for i, spot in enumerate(spots):
                if spot.contains(plate_pol) and (last_ids[i] == -1 or idx != last_ids[i]):
                    spot_dict[i] = img[j]
                    last_ids[i] = idx
                    car_ind_dict[i] = idx

    return spot_dict, last_ids, car_ind_dict


def draw_plate(img, out):
    for idx, info in out.items():
        boxes = info[-1]
        for box in boxes:
            cv2.rectangle(img, box[0], box[1], draw_configs['plate_color'], thickness=draw_configs['plate_thickness'])
            cv2.putText(
                img, str(idx), (int(box[0][0]), int(box[0][1] - 10)), 0, draw_configs['plate_font_scale'],
                draw_configs['plate_font_color'], draw_configs['plate_font_thickness']
            )


def box_inter_union(car, spot):
    spot = [spot[0], spot[1], spot[0] + spot[2], spot[1] + spot[3]]

    car = [car[0][0], car[0][1], car[1][0], car[1][1]]
    arr1, arr2 = np.array([car]), np.array([spot])

    top_left = np.maximum(arr1[:, :2], arr2[:, :2])  # [[x, y]]
    bottom_right = np.minimum(arr1[:, 2:], arr2[:, 2:])  # [[x, y]]
    wh = bottom_right - top_left

    intersection = wh[:, 0].clip(0) * wh[:, 1].clip(0)
    if intersection > 0:
        return 'Busy'
    return 'Free'


def NumberProcess(plate):
    if plate[:2] == "ՊՆ":
        if plate[-1] == "U":
            plate = plate[:-1] + "Ս"
        elif plate[-1] == "C" or plate[-1] == "G":
            plate = plate[:-1] + "Շ"
        elif plate[-1] == "S":
            plate = plate[:-1] + "Տ"
        return plate

    RusFormats = [[1, 0, 0, 0, 1, 1, 0, 0],  # 0-Digit, 1-letter
                  [1, 0, 0, 0, 1, 1, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 1, 0, 0],
                  [1, 1, 0, 0, 0, 0, 0],
                  [1, 1, 0, 0, 0, 0, 0, 0],
                  [1, 1, 0, 0, 0, 1, 0, 0],
                  [1, 1, 0, 0, 0, 1, 0, 0, 0],
                  [1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 1, 0, 0],
                  [[0, 0, 0, 0, 1, 1, 0, 0, 0]]]

    curFormat = []
    for elem in plate:
        if elem.isalpha():
            curFormat.append(1)
        else:
            curFormat.append(0)

    if curFormat in RusFormats:

        RusString = ""

        for elem in plate:
            if elem.isalpha():
                if (elem == 'A'):
                    RusString += "А"

                if (elem == 'B'):
                    RusString += "В"

                if (elem == 'E'):
                    RusString += "Е"

                if (elem == 'K'):
                    RusString += "К"

                if (elem == 'M'):
                    RusString += 'М'

                if (elem == 'H'):
                    RusString += 'Н'

                if (elem == 'P'):
                    RusString += 'Р'

                if (elem == 'C'):
                    RusString += 'С'

                if (elem == 'O'):
                    RusString += 'О'

                if (elem == 'T'):
                    RusString += 'Т'

                if (elem == 'Y'):
                    RusString += 'У'

                if (elem == 'X'):
                    RusString += 'Х'
            else:
                RusString += elem

        return RusString
    else:
        return plate
