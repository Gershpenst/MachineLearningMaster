from imutils import face_utils
import dlib
import cv2
import sys

from faceImageSearchYolo import faceImageSearchYolo
from predictFaces import PredictionPeople

class cameraPrediction:
    def __init__(self):
        self.faceDetectYolo = faceImageSearchYolo()
        self.cap = cv2.VideoCapture(0)
        self.listClass = ["Bill Gates",
        "Brad Pitt", "Donald Trump",
        "jacques chirac", "jean lassalle",
        "Jean pierre coffe",
        "Jennifer lopez", "Marine lepen",
        "Tom cruise"]

        self.predictP = PredictionPeople("./recognize_vgg16.h5", self.listClass)

    def camera(self):
        while(True):
            # self.capture frame-by-frame
            ret, frame = self.cap.read()

            allFrame, allRect = self.faceDetectYolo.detectFaces(frame)

            if(allFrame is not None):
                for f, r in zip(allFrame, allRect):

                    height, width, channels = f.shape
                    if(height == 224 and width == 224 and channels == 3):
                        recognitionStr = self.predictP.predict(f)
                        frame = cv2.rectangle(frame, r[0], r[1], (0, 0, 255), 5)
                        frame = cv2.putText(
                        frame, recognitionStr, r[0], cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the self.capture
        self.cap.release()
        cv2.destroyAllWindows()

    def image(self, img):
        frame = cv2.imread(img)
        allFrame, allRect = self.faceDetectYolo.detectFaces(frame)

        if(allFrame is not None):
            for f, r in zip(allFrame, allRect):

                height, width, channels = f.shape
                if(height == 224 and width == 224 and channels == 3):
                    recognitionStr = self.predictP.predict(f)
                    frame = cv2.rectangle(frame, r[0], r[1], (0, 0, 255), 5)
                    frame = cv2.putText(
                    frame, recognitionStr, r[0], cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

        cv2.imshow('frame', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cp = cameraPrediction()
    if(len(sys.argv) == 1):
        cp.camera()
    else:
        cp.image(sys.argv[1])
