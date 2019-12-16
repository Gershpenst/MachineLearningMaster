import cv2
from darkflow.net.build import TFNet
import numpy as np
import time


class faceImageSearchYolo:
    def __init__(self):
        self.options = {
            'model': 'tiny-yolo-1c.cfg',
            'load': 15375,
            'threshold': 0.3,
            'gpu': 1.0
        }

        self.tfnet = TFNet(self.options)
        self.colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]


    def detectFaces(self, frame, shapeVGG=224):
        allFrame = []
        allRect = []
        results = self.tfnet.return_predict(frame)
        for color, result in zip(self.colors, results):
            
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])

            perfectSquare = 0

            if(br[1]-tl[1] > br[0] - tl[0]):
                perfectSquare = br[1]-tl[1]
            else:
                perfectSquare = br[0] - tl[0]

            cropped = frame[tl[1]:tl[1]+perfectSquare , tl[0]:tl[0]+perfectSquare]
            cropped = cv2.resize(cropped,(shapeVGG,shapeVGG))

            allFrame.append(cropped)
            allRect.append((tl, br))

        
        if(allFrame):
            return allFrame, allRect

        return None, None
 
