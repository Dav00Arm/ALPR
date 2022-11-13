import os
import json
from draw_spots import SpotDrawing
import time 
from licut import *
from datetime import datetime
from crftour import crop_lines
import torch.backends.cudnn as cudnn
from csm import VideoCaptureThreading
from esycor import test_ocr
import pandas as pd
from dtc import  car_detection_yolo_one_id
from tuls import *
from cfour import *
from screeninfo import get_monitors
from copy import deepcopy
import prcsng
import nnmatch
from dtcin import Detection
from trckpi import Tracker
import detgenr as gdet
import matplotlib.pyplot as plt
from getmac import get_mac_address as gma
import sys
from shapely.geometry import Polygon
import cv2

cudnn.benchmark = True
cudnn.deterministic = True

max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 1.0
model_filename = './deepsort_models/mars-small128.pb'
spot_config = 'spot_config.json'
encoders = []
metrices = []
trackers = []

if user_configs['save_csv']:
    df = pd.DataFrame({'Path': [], 'Time': [], 'Plate':[], 'Confidence': []})
    df.to_csv(user_configs['csv_name']+'.csv', index=False)

monitor = get_monitors()[0]
width, height = monitor.width, monitor.height 
video_captures = []
prev_num = ''
gg = 0
bbboxes = []
threads = []
not_draw = True
if not os.path.isfile(spot_config):
    not_draw = False
else:
    with open(spot_config, 'r') as f:
        bbboxes = json.load(f)

for i,ip_camera in enumerate(camera_urls):
    # cap = VideoCaptureThreading(ip_camera)
    cap = VideoCaptureThreading("/home/user/Desktop/ALPR_cpp/square_car.jpg")
    cap.start()
    threads.append(cap)
    video_captures.append(cap.generator())
    if not not_draw:
        bbboxes.append([])
        frame = cap.frame 
        for j in range(spots_per_camera[i]):
            ui = SpotDrawing(frame,'Image')
            bbbox1 = ui.run()
            bbboxes[i].append(bbbox1)
            cv2.line(frame,bbbox1[0],bbbox1[1],(255,0,84),thickness=2)
            cv2.line(frame,bbbox1[1],bbbox1[2],(255,0,84),thickness=2)
            cv2.line(frame,bbbox1[2],bbbox1[3],(255,0,84),thickness=2)
            cv2.line(frame,bbbox1[3],bbbox1[0],(255,0,84),thickness=2)

        jsonFile = open(spot_config, "w+")
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
# for thr in threads:
#     if thr.stopping == True:
#         emergency_stop = True
#         thr.stop()
# if emergency_stop:
#     sys.exit()

for j in range(len(camera_urls)):
    bbbox1 = bbboxes[j]
    metrices.append(nnmatch.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget))
    trackers.append(Tracker(metrices[j]))
    encoders.append(gdet.create_box_encoder(model_filename, batch_size=8))
    spots.append([])
    status.append([])
    fix_times.append([])
    last_ln.append([])
    responses.append([])
    for k,bbbox2 in enumerate(bbbox1):
        x1y1,x2y2,x3y3,x4y4 = [i for i in bbbox2]
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
# pix = 0
start = time.time()
wait_time = 0
while True:
    # for frames in zip(*video_captures):
    frames = [cv2.imread("/home/user/Desktop/ALPR_cpp/square_car.jpg")]
    # print("Started")
    do_break = False
    base_frames = list(frames)
    ill_frames = deepcopy(list(frames))
    frames = []
    for j in range(len(camera_urls)):
        bbbox1 = bbboxes[j]
        frames.append(base_frames[j])
        for k,bbbox2 in enumerate(bbbox1):
            cv2.line(ill_frames[j],tuple(bbbox2[1]),tuple(bbbox2[2]),(255,0,84),thickness=2)
            cv2.line(ill_frames[j],tuple(bbbox2[0]),tuple(bbbox2[1]),(255,0,84),thickness=2)
            cv2.line(ill_frames[j],tuple(bbbox2[2]),tuple(bbbox2[3]),(255,0,84),thickness=2)
            cv2.line(ill_frames[j],tuple(bbbox2[3]),tuple(bbbox2[0]),(255,0,84),thickness=2)
            if user_configs['display_spot_id']:
                cv2.putText(ill_frames[j], f'Spot ID {k}', (bbbox2[0][0]+5, bbbox2[1][0]+20), cv2.LINE_AA, 0.5, (0,0,0), 2)
        text = ''
        if user_configs['display_date']:
            text += datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if user_configs['display_camera_id']:
            text += f' Cam ID {j}'
        cv2.putText(ill_frames[j], text, (10, 20), cv2.LINE_AA, 0.8, (0,0,0), 2)

    for cam_id, camera_frame in enumerate(frames):
        last_ids = last_idss[cam_id]
        car_images,draw_boxes_car, scores, classes = car_detection_yolo_one_id(camera_frame)
        bboxes = []
        for box in draw_boxes_car:
            bboxes.append(convert_xywh(box))
        features = encoders[cam_id](frames[cam_id],bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, classes, features)]
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = prcsng.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices] 
        trackers[cam_id].predict()
        trackers[cam_id].update(detections)
        cam_images = {}
        ids = []
        boxes = []
        for track in trackers[cam_id].tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            maximum_y,maximum_x,_ = frames[cam_id].shape
            x_min = int(max(0,bbox[0]))
            x_max = int(min(bbox[2],maximum_x))
            y_min = int(max(0,bbox[1]))
            y_max = int(min(bbox[3],maximum_y))
            boxes.append([(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3]))])
            cam_images[track.track_id] = frames[cam_id][y_min:y_max,x_min:x_max]
            ids.append(track.track_id)

        out, car_ids = detect_plate_onnx_id(cam_images, car_boxes=boxes)
        
        spot_dict, last_ids, current_spot_dict = check_box(spots[cam_id], out, last_ids)
        for key, value in current_spot_dict.items():
            status[cam_id][key] = value
            if len(changed) > 0:
                if changed[f'{cam_id}_{key}'] == 'Busy':
                    cv2.line(ill_frames[cam_id],tuple(bbboxes[cam_id][key][1]),tuple(bbboxes[cam_id][key][2]),(0,0,255),thickness=2)
                    cv2.line(ill_frames[cam_id],tuple(bbboxes[cam_id][key][0]),tuple(bbboxes[cam_id][key][1]),(0,0,255),thickness=2)
                    cv2.line(ill_frames[cam_id],tuple(bbboxes[cam_id][key][2]),tuple(bbboxes[cam_id][key][3]),(0,0,255),thickness=2)
                    cv2.line(ill_frames[cam_id],tuple(bbboxes[cam_id][key][3]),tuple(bbboxes[cam_id][key][0]),(0,0,255),thickness=2)

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
                    color = (0,0,255)

                elif time.time() - fix_times[spot_times][spot_time] - wait_time > 10:
                    changed[f'{spot_times}_{spot_time}'] = 'Free'
                    color = (255,0,0)


        draw_plate(ill_frames[cam_id],out)
        for spot_id, img in spot_dict.items():
            spot = spots[cam_id][spot_id]
            if img is not None :
                number_images = crop_lines([img])
                if len(number_images)>0:    
                    for nm_img_id,res in enumerate(number_images): 
                        date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        prediction , conf = test_ocr(res)
                    if len(prediction) <= 3 or (len(prediction) == 4 and prediction[-2].isalpha()) or (len(prediction) == 6 and prediction[-2].isnumeric()):
                        last_ids[spot_id] = -1
                    elif prediction != last_ln[cam_id][spot_id] and conf >= 97:
                        # if pix > 10:
                        responses[cam_id][spot_id] = True #send_data(cam_images[last_ids[spot_id]], img, cam_id, prediction, date_time, conf, spot_id)
                        print(prediction)
                        last_ln[cam_id][spot_id] = prediction

                    elif conf < 97:
                        # print(prediction)
                        last_ids[spot_id] = -1
    wait_time = 0
    if prev_changed != changed:
        frees = {}
        frees_spot = {}
        for key, _ in changed.items():
            if changed[key] == 'Free':
                if len(prev_changed) > 0 and (prev_changed[key] != 'Free' and responses[int(key.split('_')[0])][int(key.split('_')[1])]):            
                    frees['mac_address'] = mac_addresses[int(key.split('_')[0])]
                    frees_spot[int(key.split('_')[1])] = 'Free'
                    frees['time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    frees['spot'] = frees_spot
                    # print(frees)
                    # if pix > 10:
                    #post_data(locator_url, frees, name='places')
                    changed_free[key] = 0
                    responses[int(key.split('_')[0])][int(key.split('_')[1])] = False

                if len(prev_changed) == 0 :
                    print(status)
                    if status[int(key.split('_')[0])][int(key.split('_')[1])] == 'Free':
                        frees['mac_address'] = mac_addresses[int(key.split('_')[0])]
                        frees_spot[int(key.split('_')[1])] = 'Free'
                        frees['time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        frees['spot'] = frees_spot
                        # print(frees)
                        #if pix > 10:
                        #post_data(locator_url, frees, name='places')
                        changed_free[key] = 0
                        responses[int(key.split('_')[0])][int(key.split('_')[1])] = False

                    
            
        prev_changed = deepcopy(changed)
    
    out_frame = show_images(ill_frames,width, height)
    cv2.imshow("Image",out_frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'): 
        cv2.destroyAllWindows()

        sys.exit()


    wait_time = time.time() - start_time
    

        