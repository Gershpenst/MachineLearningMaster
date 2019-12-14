from imutils import face_utils
import dlib
import cv2

from faceImageSearchYolo import faceImageSearchYolo
from predictFaces import PredictionPeople

listClass = ['ali',
            'other', "Bill Gates",
            "Brad Pitt", "Donald Trump",
            "jacques chirac", "jean lassalle",
            "Jean pierre coffe",
            "Jennifer lopez", "Marine lepen",
            "Tom cruise"]

predictP = PredictionPeople("./recognize_4class_1_0.5dropoutPlus.h5", listClass)

if __name__ == "__main__":
    faceDetectYolo = faceImageSearchYolo()
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        allFrame, allRect = faceDetectYolo.detectFaces(frame)

        if(allFrame is not None):
            for f, r in zip(allFrame, allRect):

                height, width, channels = f.shape
                if(height == 224 and width == 224 and channels == 3):
                    recognitionStr = predictP.predict(f)
                    frame = cv2.rectangle(frame, r[0], r[1], (0, 0, 255), 5)
                    frame = cv2.putText(
                        frame, recognitionStr, r[0], cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
