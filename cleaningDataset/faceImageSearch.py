# 
#

import os
from imutils import face_utils
import dlib
import cv2
from imutils.face_utils import FaceAligner
 
class faceImageSearch():
    def __init__(self, shape=256):
        # initialise la librairie dlib afin de trouver un visage
        self._shape_predictor = "shape_predictor_68_face_landmarks.dat"
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(self._shape_predictor)

        # Initialisation de la classe permettant de mettre droit un visage
        self._face_aligner = FaceAligner(self._predictor, desiredFaceWidth=shape)


    def detectVisage(self, strImg, path='', dirname='', incr=0):
        # lit et convertit l'image en noir et blanc
        image = cv2.imread(strImg)

        # Si l'image n'est pas lisible, quitter la fonction
        if(image is None):
            print("Erreur avec l'image: ", strImg)
            return incr

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detecte les visages dans l'image
        rects = self._detector(gray, 0)

        # loop over the face detections
        for (i, rect) in enumerate(rects):

            # Determine la position du visage, des yeux, de la bouche, du nez ...
            shape = self._predictor(gray, rect)

            # Aligne par rapport aux yeux, le viosage humain (si le visage est penché) et rétrécit la ROI
            faceAligned = self._face_aligner.align(image, gray, rect)
            
            if(path != ''):
                incr = incr + 1
                print("   |-----", incr, " ", strImg)
                cv2.imwrite(os.path.join(path, dirname+str(incr)+".jpg"), faceAligned)

        return incr