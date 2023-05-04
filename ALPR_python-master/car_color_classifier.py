import cv2
import numpy as np


class CarColorClassifier:
    # This class classifies car's color with opencv, supports 12 color classification

    def __init__(self):
        self.colors = [
            "white", "black", "brown", "gray", "orange", "yellow", "green", "cyan", "blue", "purple", "pink", "red"
        ]
        self.range_lower_bounds = [
            (0, 0, 160), (0, 0, 0), (0, 40, 56), (0, 0, 56), (6, 86, 86), (21, 46, 56),(36, 46, 56), (88, 46, 56),
            (103, 46, 86), (127, 46, 56), (152, 46, 56), (0, 46, 56), (173, 86, 56)
        ]
        self.range_upper_bounds = [
            (179, 45, 255), (179, 255, 55), (65, 224, 185), (179, 45, 200), (20, 255, 255), (35, 255, 255),
            (87, 255, 255), (102, 255, 255), (131, 255, 255), (157, 255, 255), (176, 210, 255), (5, 255, 255),
            (179, 255, 255)
        ]

    def __call__(self, car_images, car_ind_dict, crop_size_up=0.4, crop_size_down=0.1, crop_size_left=0.1,
                 crop_size_right=0.1, *args, **kwargs):
        predicted_colors = {}
        for spot_id, car_id in car_ind_dict.items():
            hsv_image = cv2.cvtColor(car_images[car_id], cv2.COLOR_BGR2HSV)

            masks = self.get_masks(hsv_image)
            cropped_masks = self.crope_masks(masks, crop_size_up, crop_size_down, crop_size_left, crop_size_right)

            predicted_color = self.get_color(cropped_masks)
            predicted_colors[spot_id] = predicted_color

        return predicted_colors

    def get_masks(self, hsv_image):
        masks = []
        for color_id in range(len(self.range_lower_bounds) - 1):
            range_lower = self.range_lower_bounds[color_id]
            range_upper = self.range_upper_bounds[color_id]

            mask = cv2.inRange(hsv_image, range_lower, range_upper)

            if color_id == len(self.range_lower_bounds) - 2:
                range_lower2 = self.range_lower_bounds[color_id + 1]
                range_upper2 = self.range_upper_bounds[color_id + 1]

                mask2 = cv2.inRange(hsv_image, range_lower2, range_upper2)
                mask = cv2.bitwise_or(mask, mask2)

            masks.append(mask)

        return masks

    def crope_masks(self, masks, crop_size_up, crop_size_down, crop_size_left, crop_size_right):
        y_start = int(crop_size_up * masks[0].shape[0])
        y_end = masks[0].shape[0] - int(crop_size_down * masks[0].shape[0])

        x_start = int(crop_size_left * masks[0].shape[1])
        x_end = masks[0].shape[1] - int(crop_size_right * masks[0].shape[1])

        cropped_masks = []
        for mask in masks:
            cropped_mask = mask[y_start: y_end, x_start: x_end]
            cropped_masks.append(cropped_mask)

        return cropped_masks

    def get_color(self, cropped_masks):
        cnts = np.zeros(len(cropped_masks))
        for i, cropped_mask in enumerate(cropped_masks):
            cnts[i] = np.sum(cropped_mask)

        return self.colors[np.argmax(cnts)]
