import os
import json
from draw_spots import SpotDrawing
from license_plate_utils import *
from PROJECT_Refiner_Model_SOFTWARE_AI.functions import crop_lines
import torch.backends.cudnn as cudnn
from VideoCapture import VideoCaptureThreading
from PROJECT_Optical_Character_Recognition_Model_SOFTWARE_AI.inference import test_ocr
from PROJECT_Car_Detection_Model_SOFTWARE_AI.inference import car_detection_yolo_one_id
from utils import *
from screeninfo import get_monitors
from copy import deepcopy
import tracker.nms
from tracker import distance
from tracker.detection import Detection
from tracker.tracker import Tracker
import tracker.detection_generator as gdet
import matplotlib.pyplot as plt
import sys
from shapely.geometry import Polygon
import cv2
from datetime import datetime
from configs.general import main_configs, draw_configs, camera_configs
from PROJECT_Car_Detection_Model_SOFTWARE_AI.car_det_configs import model_configs
from car_color_classifier import CarColorClassifier

cudnn.benchmark = True
cudnn.deterministic = True


if __name__ == '__main__':
    encoders = []
    metrices = []
    trackers = []

    if not os.path.isdir(main_configs['logdir']):
        os.mkdir(main_configs['logdir'])

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
        metrices.append(distance.NearestNeighborDistanceMetric(
            main_configs['nn_distance_metric'],
            main_configs['max_cosine_distance'],
            main_configs['nn_budget']
        ))
        trackers.append(Tracker(metrices[j]))
        encoders.append(gdet.create_box_encoder(main_configs['model_filename'], batch_size=main_configs['batch_size']))
        spots.append([])
        last_ln.append([])
        responses.append([])
        for k, bbbox2 in enumerate(bbbox1):
            x1y1, x2y2, x3y3, x4y4 = [i for i in bbbox2]
            spots[j].append(Polygon([x1y1, x2y2, x3y3, x4y4]))
            responses[j].append(False)
            last_ln[j].append(None)

        last_idss.append(np.negative(np.ones((len(spots[j])))))

    counter = 0
    do_update = True
    start_time = time.time()
    start = time.time()
    wait_time = 0
    last_req_time = 0
    color_classifier = CarColorClassifier()
    car_log_index = 0
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
                indices = tracker.nms.non_max_suppression(boxs, classes, main_configs['nms_max_overlap'], scores)
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

                spot_dict, last_ids, car_ind_dict = check_box(spots[cam_id], out, last_ids)

                car_colors_dict = color_classifier(cam_images, car_ind_dict)

                draw_plate(ill_frames[cam_id], out)
                for spot_id, img in spot_dict.items():
                    cls = labels[car_ind_dict[spot_id]]
                    label = model_configs['class_names'][cls]
                    color = car_colors_dict[spot_id]

                    os.mkdir(f"{main_configs['logdir']}/car{car_log_index}")
                    cv2.imwrite(f"{main_configs['logdir']}/car{car_log_index}/{label}_{color}.jpg", cam_images[car_ind_dict[spot_id]])
                    spot = spots[cam_id][spot_id]
                    if img is not None:
                        cv2.imwrite(f"{main_configs['logdir']}/car{car_log_index}/plate.jpg", img)
                        number_images = crop_lines([img])  # Refiner part(CRAFT)
                        if len(number_images) > 0:
                            last_ids[spot_id] = -1
                            for nm_img_id, res in enumerate(number_images):
                                date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                prediction, conf = test_ocr(res)  # OCR part
                                prediction = NumberProcess(prediction)
                                for i in range(len(res)):
                                    cv2.imwrite(f"{main_configs['logdir']}/car{car_log_index}/line{nm_img_id}.jpg", np.array(res[i]))
                                # print(prediction, conf)
                                cls = labels[car_ind_dict[spot_id]]
                                label = model_configs['class_names'][cls]
                                color = car_colors_dict[spot_id]
                                f = open(f"{main_configs['logdir']}/car{car_log_index}/{prediction}_{conf}.txt", "x")
                                if len(prediction) <= 3 or (len(prediction) == 4 and prediction[-2].isalpha()) or (
                                        len(prediction) == 6 and prediction[-2].isnumeric()):
                                    last_ids[spot_id] = -1
                                elif conf >= main_configs['ocr_conf_threshold']:  # prediction != last_ln[cam_id][spot_id]
                                    responses[cam_id][spot_id] = True  # send_data(cam_images[last_ids[spot_id]], img, cam_id, prediction, date_time, conf, spot_id)

                                    print("FINAL:", prediction, conf, label, color)
                                    if last_req_time == 0 or time.time() - last_req_time >= main_configs['request_timeout']:
                                        print('Requesting')
                                        # request_to_barrier(barrier_configs['barrier_open_url'])
                                        last_req_time = time.time()
                                    else:
                                        print('Dont send request')

                                    last_ln[cam_id][spot_id] = prediction

                                elif conf < main_configs['ocr_conf_threshold']:
                                    last_ids[spot_id] = -1
                    car_log_index+=1

            wait_time = 0
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