import os
import json
import time
from draw_spots import SpotDrawing
from licut import *
from crftour import crop_lines
import torch.backends.cudnn as cudnn
from csm import VideoCaptureThreading
from esycor import test_ocr
import pandas as pd
from dtc import car_detection_yolo_one_id
from tuls import *
from screeninfo import get_monitors
from copy import deepcopy
import prcsng
import nnmatch
from dtcin import Detection
from trckpi import Tracker
import detgenr as gdet
import matplotlib.pyplot as plt
import sys
from shapely.geometry import Polygon
import cv2
from datetime import datetime
from configs.general import main_configs, draw_configs, camera_configs, barrier_configs
from configs.model_configs import car_det_configs
from car_color_classifier import CarColorClassifier
from dtsnd import request_to_barrier
from getmac import get_mac_address as gma

cudnn.benchmark = True
cudnn.deterministic = True


if __name__ == '__main__':
    encoders = []
    metrices = []
    trackers = []

    if user_configs['save_csv']:
        df = pd.DataFrame({'Path': [], 'Time': [], 'Plate': [], 'Confidence': []})
        df.to_csv(user_configs['csv_name'] + '.csv', index=False)

    prev_num = ''
    gg = 0
    monitor = get_monitors()[0]
    width, height = monitor.width, monitor.height
    video_captures = []
    bbboxes = []
    threads = []
    not_draw = True
    wl = load_whiteList(main_configs['wl_path'])
    if not os.path.isfile(main_configs['spot_config']):
        not_draw = False
    else:
        with open(main_configs['spot_config'], 'r') as f:
            bbboxes = json.load(f)

    for i, ip_camera in enumerate(camera_configs['camera_urls']):
        cap = VideoCaptureThreading(0)
        cap.start()
        threads.append(cap)
        video_captures.append(cap.generator())
        if not not_draw:
            bbboxes.append([])
            frame = cap.frame
            for j in range(camera_configs['spots_per_camera'][i]):
                ui = SpotDrawing(frame, 'Image')
                bbbox1 = ui.run()
                bbboxes[i].append(bbbox1)
                cv2.line(frame, bbbox1[0], bbbox1[1], draw_configs['spot_line_color'],
                         thickness=draw_configs['spot_line_thickness'])
                cv2.line(frame, bbbox1[1], bbbox1[2], draw_configs['spot_line_color'],
                         thickness=draw_configs['spot_line_thickness'])
                cv2.line(frame, bbbox1[2], bbbox1[3], draw_configs['spot_line_color'],
                         thickness=draw_configs['spot_line_thickness'])
                cv2.line(frame, bbbox1[3], bbbox1[0], draw_configs['spot_line_color'],
                         thickness=draw_configs['spot_line_thickness'])

            jsonFile = open(main_configs['spot_config'], "w+")
            jsonFile.write(json.dumps(bbboxes))
            jsonFile.close()


    last_idss = []
    spots = []
    responses = []
    status = []
    fix_times = []
    changed = {}
    changed_free = {}
    last_ln = []
    emergency_stop = False
    # if mc_address[0] != gma():
    #     print("Package is not installed")
    #     sys.exit()
    for thr in threads:
        if thr.stopping:
            emergency_stop = True
            thr.stop()
    if emergency_stop:
        sys.exit()

    for j in range(len(camera_configs['camera_urls'])):
        bbbox1 = bbboxes[j]
        metrices.append(nnmatch.NearestNeighborDistanceMetric(
            main_configs['nn_distance_metric'],
            main_configs['max_cosine_distance'],
            main_configs['nn_budget']
        ))
        trackers.append(Tracker(metrices[j]))
        encoders.append(gdet.create_box_encoder(main_configs['model_filename'], batch_size=main_configs['batch_size']))
        spots.append([])
        status.append([])
        fix_times.append([])
        last_ln.append([])
        responses.append([])
        for k, bbbox2 in enumerate(bbbox1):
            x1y1, x2y2, x3y3, x4y4 = [i for i in bbbox2]
            spots[j].append(Polygon([x1y1, x2y2, x3y3, x4y4]))
            responses[j].append(False)
            status[j].append(None)
            fix_times[j].append(0)
            last_ln[j].append(None)

        last_idss.append(np.negative(np.ones((len(spots[j])))))

    counter = 0
    do_update = True
    prev_status = deepcopy(status)
    prev_changed = {}
    start_time = time.time()
    start = time.time()
    wait_time = 0
    while True:
        for frames in zip(*video_captures):
            do_break = False
            base_frames = list(frames)
            ill_frames = deepcopy(list(frames))
            frames = []
            for j in range(len(camera_configs['camera_urls'])):
                bbbox1 = bbboxes[j]
                frames.append(base_frames[j])
                for k, bbbox2 in enumerate(bbbox1):

                    cv2.line(ill_frames[j], bbbox2[1], bbbox2[2], draw_configs['spot_line_color'],
                             thickness=draw_configs['spot_line_thickness'])
                    cv2.line(ill_frames[j], bbbox2[0], bbbox2[1], draw_configs['spot_line_color'],
                             thickness=draw_configs['spot_line_thickness'])
                    cv2.line(ill_frames[j], bbbox2[2], bbbox2[3], draw_configs['spot_line_color'],
                             thickness=draw_configs['spot_line_thickness'])
                    cv2.line(ill_frames[j], bbbox2[3], bbbox2[0], draw_configs['spot_line_color'],
                             thickness=draw_configs['spot_line_thickness'])
                    if user_configs['display_spot_id']:
                        cv2.putText(
                            ill_frames[j], f'Spot ID {k}', (bbbox2[0][0] + 5, bbbox2[1][0] + 20), cv2.LINE_AA,
                            draw_configs['spot_font_scale'], draw_configs['spot_font_color'],
                            draw_configs['spot_font_thickness']
                        )
                text = ''
                if user_configs['display_date']:
                    text += datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                if user_configs['display_camera_id']:
                    text += f' Cam ID {j}'
                cv2.putText(
                    ill_frames[j], text, draw_configs['time_org_coordinates'], cv2.LINE_AA,
                    draw_configs['time_font_scale'], draw_configs['time_font_color'],
                    draw_configs['time_font_thickness']
                )

            for cam_id, camera_frame in enumerate(frames):
                last_ids = last_idss[cam_id]
                car_images, draw_boxes_car, scores, classes = car_detection_yolo_one_id(camera_frame)
                bboxes = []
                for box in draw_boxes_car:
                    bboxes.append(convert_xywh(box))
                features = encoders[cam_id](frames[cam_id], bboxes)
                detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                              zip(bboxes, scores, classes, features)]
                cmap = plt.get_cmap('tab20b')
                colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
                boxs = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                classes = np.array([d.class_name.to('cpu') for d in detections])
                indices = prcsng.non_max_suppression(boxs, classes, main_configs['nms_max_overlap'], scores)
                detections = [detections[i] for i in indices]
                trackers[cam_id].predict()
                trackers[cam_id].update(detections)
                cam_images = {}
                labels = {}
                boxes = []
                for track in trackers[cam_id].tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue

                    label = track.get_class().item()
                    bbox = track.to_tlbr()
                    color = colors[int(track.track_id) % len(colors)]
                    color = [i * 255 for i in color]
                    maximum_y, maximum_x, _ = frames[cam_id].shape
                    x_min = int(max(0, bbox[0]))
                    x_max = int(min(bbox[2], maximum_x))
                    y_min = int(max(0, bbox[1]))
                    y_max = int(min(bbox[3], maximum_y))
                    boxes.append([(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))])
                    cam_images[track.track_id] = frames[cam_id][y_min: y_max, x_min: x_max]
                    labels[track.track_id] = label

                out, car_ids = detect_plate_onnx_id(cam_images, car_boxes=boxes)

                spot_dict, last_ids, current_spot_dict, car_ind_dict = check_box(spots[cam_id], out, last_ids)

                color_classifier = CarColorClassifier()
                car_colors_dict = color_classifier(cam_images, car_ind_dict)

                for key, value in current_spot_dict.items():
                    status[cam_id][key] = value
                    if len(changed) > 0:
                        if changed[f'{cam_id}_{key}'] == 'Busy':
                            cv2.line(ill_frames[cam_id], bbboxes[cam_id][key][1], bbboxes[cam_id][key][2],
                                     draw_configs['status_line_color'], thickness=draw_configs['status_line_thickness'])
                            cv2.line(ill_frames[cam_id], bbboxes[cam_id][key][0], bbboxes[cam_id][key][1],
                                     draw_configs['status_line_color'], thickness=draw_configs['status_line_thickness'])
                            cv2.line(ill_frames[cam_id], bbboxes[cam_id][key][2], bbboxes[cam_id][key][3],
                                     draw_configs['status_line_color'], thickness=draw_configs['status_line_thickness'])
                            cv2.line(ill_frames[cam_id], bbboxes[cam_id][key][3], bbboxes[cam_id][key][0],
                                     draw_configs['status_line_color'], thickness=draw_configs['status_line_thickness'])

                if prev_status != status:
                    for spots_i in range(len(status)):
                        for spot_i in range(len(status[spots_i])):
                            if prev_status[spots_i][spot_i] == 'Busy' and status[spots_i][spot_i] == 'Free':
                                fix_times[spots_i][spot_i] = time.time()
                            elif prev_status[spots_i][spot_i] == 'Free' and status[spots_i][spot_i] == 'Busy':
                                fix_times[spots_i][spot_i] = -1
                    prev_status = deepcopy(status)

                for spot_times in range(len(fix_times)):
                    for spot_time in range(len(fix_times[spot_times])):
                        if fix_times[spot_times][spot_time] == -1:
                            changed[f'{spot_times}_{spot_time}'] = 'Busy'
                            color = draw_configs['status_busy_color']

                        elif time.time() - fix_times[spot_times][spot_time] - wait_time > 10:
                            changed[f'{spot_times}_{spot_time}'] = 'Free'
                            color = draw_configs['status_free_color']

                draw_plate(ill_frames[cam_id], out)
                for spot_id, img in spot_dict.items():
                    spot = spots[cam_id][spot_id]
                    if img is not None:
                        number_images = crop_lines([img])
                        if len(number_images) > 0:
                            last_ids[spot_id] = -1
                            for nm_img_id, res in enumerate(number_images):
                                date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                prediction, conf = test_ocr(res)
                                prediction = NumberProcess(prediction)
                                cls = labels[car_ind_dict[spot_id]]
                                label = car_det_configs['class_names'][cls]
                                color = car_colors_dict[spot_id]
                                if len(prediction) <= 3 or (len(prediction) == 4 and prediction[-2].isalpha()) or (
                                        len(prediction) == 6 and prediction[-2].isnumeric()):
                                    last_ids[spot_id] = -1
                                elif (prediction != last_ln[cam_id][spot_id]) and (
                                        conf >= main_configs['ocr_conf_threshold']):# and (prediction not in wl):
                                    # if pix > 10:
                                    responses[cam_id][spot_id] = True  # send_data(cam_images[last_ids[spot_id]], img, cam_id, prediction, date_time, conf, spot_id)

                                    print("FINAL:", prediction, conf, label, color)
                                    # request_to_barrier(barrier_configs['barrier_open_url'])

                                    last_ln[cam_id][spot_id] = prediction

                                elif conf < main_configs['ocr_conf_threshold']:
                                    # print(prediction)
                                    last_ids[spot_id] = -1
            wait_time = 0
            if prev_changed != changed:
                frees = {}
                frees_spot = {}
                for key, _ in changed.items():
                    if changed[key] == 'Free':
                        if len(prev_changed) > 0 and (prev_changed[key] != 'Free' and responses[int(key.split('_')[0])][int(key.split('_')[1])]):
                            frees['mac_address'] = camera_configs['mac_addresses'][int(key.split('_')[0])]
                            frees_spot[int(key.split('_')[1])] = 'Free'
                            frees['time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            frees['spot'] = frees_spot
                            # print(frees)
                            # if pix > 10:
                            # post_data(locator_url, frees, name='places')
                            changed_free[key] = 0
                            responses[int(key.split('_')[0])][int(key.split('_')[1])] = False

                        if len(prev_changed) == 0:
                            # print(status)
                            if status[int(key.split('_')[0])][int(key.split('_')[1])] == 'Free':
                                frees['mac_address'] = camera_configs['mac_addresses'][int(key.split('_')[0])]
                                frees_spot[int(key.split('_')[1])] = 'Free'
                                frees['time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                frees['spot'] = frees_spot
                                # print(frees)
                                # if pix > 10:
                                # post_data(locator_url, frees, name='places')
                                changed_free[key] = 0
                                responses[int(key.split('_')[0])][int(key.split('_')[1])] = False

                prev_changed = deepcopy(changed)
            # print("Inference time",time.time() - dv)
            out_frame = show_images(ill_frames, width, height)
            cv2.imshow("Image", out_frame)

            key = cv2.waitKey(1)
            # for thr in threads:
            #     if thr.stopping == True:
            #         emergency_stop = True
            # if time.time() - start_time > 100:
            #     print("Reinitilize")
            #     for thr in threads:
            #         thr.stop()
            #     del threads
            #     del video_captures
            #     threads = []
            #     video_captures = []
            #     for ip in camera_urls:
            #         cap = VideoCaptureThreading(ip)
            #         cap.start()
            #         threads.append(cap)
            #         video_captures.append(cap.generator())
            #     start_time = time.time()
            #     break
            if key == ord('q'):  # or emergency_stop:
                # cap = VideoCaptureThreading(ip_camera)
                # cap.start()
                # for thr in threads:
                #     thr.stop()
                cv2.destroyAllWindows()
                break

        wait_time = time.time() - start_time