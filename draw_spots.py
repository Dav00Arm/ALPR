import cv2
import numpy as np
class SpotDrawing:
    def __init__(self,image,window_name):
        self.image = image
        self.window_name = window_name
        self.do_break = False
        self.ill_image = ()
        self.coords = []
        self.spots = []
        self.line_coords = []
        self.current_drawing = []
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name,self.MouseCallback)

    def MouseCallback(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.coords) != 4:
                self.coords.append((x,y))
                if len(self.coords) == 4:
                    self.spots.append(self.coords)
            
        if event == cv2.EVENT_MOUSEMOVE and len(self.coords) != 4:
            if len(self.coords) > 0:
                self.line_coords = self.coords[-1],(x,y)
                self.current_drawing = self.coords[0], (x,y)

    def run(self):
        while True:
            if len(self.line_coords) > 0:
                self.ill_image = self.image.copy()
            if isinstance(self.ill_image,tuple):
                self.show_image(self.image)
                self.always_draw(self.image)
            else:
                self.draw(self.ill_image)
                self.show_image(self.ill_image)
                self.always_draw(self.ill_image)
            self.ill_image = ()
            if len(self.spots) == 1:
                return self.spots[0]
                

    def draw(self,img):
        if len(self.line_coords) > 0:
            if len(self.coords) >= 2:
                first = self.coords[0]
                for pt in self.coords[1:]:
                    cv2.line(img,first,pt,(255,0,255),thickness=2)
                    first = pt
            cv2.line(img,self.line_coords[0],self.line_coords[1],(255,0,255),thickness=2)
            cv2.line(img,self.current_drawing[0],self.current_drawing[1],(255,0,255),thickness=2)

    def always_draw(self,image):
        if len(self.spots) > 0:
            for line in self.spots:
                first = line[0]
                for pt in line[1:]:   
                    cv2.line(image,first,pt,(0,0,255),thickness=2)
                    first = pt
                cv2.line(image,line[0],first,(0,0,255),thickness=2)


    def show_image(self,image):
        key = cv2.waitKey(22)
        cv2.imshow(self.window_name,image)
