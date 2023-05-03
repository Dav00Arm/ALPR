from tracemalloc import start
import cv2
import os
from licut import *
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

cudnn.benchmark = True
cudnn.deterministic = True

max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 1.0
model_filename = './deepsort_models/mars-small128.pb'
encoders = []
metrices = []
trackers = []
# if mc_address[0] != gma():
#     print("Package is not installed")
#     sys.exit()

if user_configs['save_csv']:
    df = pd.DataFrame({'Path': [], 'Time': [], 'Plate':[], 'Confidence': []})
    df.to_csv(user_configs['csv_name']+'.csv', index=False)

monitor = get_monitors()[0]
width, height = monitor.width, monitor.height 
video_captures = []
prev_num = ''
gg = 0
bbboxes = []
prev_status = {}
q = 0
for i,ip_camera in enumerate(camera_urls):
    cap = VideoCaptureThreading(ip_camera)
    video_captures.append(cap.start().generator())
    bbboxes.append([])
    frame = cap.frame 
    for j in range(spots_per_camera[i]):
        bbbox1 = cv2.selectROI("Image", frame)
        bbboxes[i].append(bbbox1)
        cv2.rectangle(frame,(bbboxes[i][j][0],bbboxes[i][j][1]),(bbboxes[i][j][0]+bbboxes[i][j][2],bbboxes[i][j][1] + bbboxes[i][j][3]),(255,0,84))

last_idss = []
spots = []
status = {}
responses = []
last_ln = []
spots_status = []
for j,bbbox1 in enumerate(bbboxes):
    metrices.append(nnmatch.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget))
    trackers.append(Tracker(metrices[j]))
    encoders.append(gdet.create_box_encoder(model_filename, batch_size=8))
    spots.append([])
    last_ln.append([])
    spots_status.append([])
    responses.append([])
    for k,bbbox2 in enumerate(bbbox1):
        x,y,w,h = [int(i) for i in bbbox2]
        spots[j].append([x,y,w,h])
        responses[j].append(False)
        last_ln[j].append(None)
        spots_status[j].append('Free')
    last_idss.append(np.negative(np.ones((len(spots[j])))))

counter = 0
for frames in zip(*video_captures):
    # time.sleep(3)
    # if counter % 100000 == 0:
    do_break = False
    base_frames = list(frames)
    ill_frames = deepcopy(list(frames))
    frames = []
    for j,bbbox1 in enumerate(bbboxes):
        frames.append(base_frames[j])
        for k,bbbox2 in enumerate(bbbox1):
            x,y,w,h = [int(i) for i in bbbox2]
            status[f'{j}_{k}'] = "Free"
            cv2.rectangle(ill_frames[j],(bbbox2[0],bbbox2[1]),(bbbox2[0]+bbbox2[2],bbbox2[1]+bbbox2[3]), (0,0,255))
            if user_configs['display_spot_id']:
                cv2.putText(ill_frames[j], f'Spot ID {k}', (bbbox2[0]+5, bbbox2[1]+20), cv2.LINE_AA, 0.5, (0,0,0), 2)
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
            cv2.putText(ill_frames[cam_id],str(track.track_id),(int(bbox[0]),int(bbox[1]-10)),0,0.75,(255,255,255),2)
    

        out, car_ids = detect_plate_onnx_id(cam_images, car_boxes=boxes)
        spot_dict, last_ids = check_box_free(spots[cam_id], out, last_ids, cam_id, status)

        draw_plate(ill_frames[cam_id],out)
        for spot_id, img in spot_dict.items():
            spot = spots[cam_id][spot_id]
            bbox = car_ids[last_ids[spot_id]]
            if img is not None:
                cv2.rectangle(ill_frames[cam_id],(spot[0],spot[1]),(spot[0]+spot[2],spot[1]+spot[3]), (0,0,255))
                number_images = crop_lines([img])
                if len(number_images)>0:    
                    for nm_img_id,res in enumerate(number_images): 
                        date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        prediction , conf = test_ocr(res)
                    responses[cam_id][spot_id] = send_data(cam_images[last_ids[spot_id]], img, cam_id, prediction, date_time, conf, spot_id)
                    last_ln[cam_id][spot_id] = prediction


    
                    # print('[{}]: Cam ID: {}  Frame ID: {}  Prediction: {}   confidence: {}%' .format(date_time,cam_id,spot_id,prediction, conf))
        for 
        for spot_id, spot in enumerate(spots[cam_id]):
            # print(cam_id)
            car_id_ = last_ids[spot_id]
            if (car_id_ != -1):
                try:
                    status[f'{cam_id}_{spot_id}'] = box_inter_union(car_ids[last_ids[spot_id]], spot)
                except KeyError:
                    pass
            if status[f'{cam_id}_{spot_id}'] == "Busy":
                cv2.rectangle(ill_frames[cam_id],(spot[0],spot[1]),(spot[0]+spot[2],spot[1]+spot[3]), (0,0,255), thickness=4)
            else:
                cv2.rectangle(ill_frames[cam_id],(spot[0],spot[1]),(spot[0]+spot[2],spot[1]+spot[3]), (145,0,255), thickness=2)

    frees = {}
    sps = {}
    for i,(key, value) in enumerate(status.items()):
        cam_id, spot_id = key.split('_')[0],key.split('_')[1]
        if value == 'Free':
            sps[spot_id] = value
        if (i != 0 and i % 2) :
            frees['mac_address'] = mac_addresses[int(cam_id)]
            frees['spot'] = sps
            frees['time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(frees)
            post_data(locator_url, frees, name='places')
            time.sleep(1)
            responses[int(cam_id)][int(spot_id)] = False
        # else:
            
        prev_cam_id = deepcopy(cam_id)
    prev_status = deepcopy(status)
    
    out_frame = show_images(ill_frames,width, height)
    cv2.imshow("Image",out_frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        cv2.destroyAllWindows()
        break
