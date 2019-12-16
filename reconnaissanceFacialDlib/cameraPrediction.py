from imutils import face_utils
import dlib
import cv2
import sys

from faceImageSearchDlib import faceImageSearchDlib
from predictFaces import PredictionPeople

class cameraPrediction:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.faceSearch = faceImageSearchDlib()
        self.listClass = ["Bill Gates",
        "Brad Pitt", "Donald Trump",
        "jacques chirac", "jean lassalle",
        "Jean pierre coffe",
        "Jennifer lopez", "Marine lepen",
        "Tom cruise"]

        # Initialisation du modéle pré-entrainer VGG-16
        self.predictP = PredictionPeople("./recognize_vgg16.h5", self.listClass)

    def camera(self):
        while True:
            recognitionStr = ""

            # Webcam
            _, image = self.cap.read()

            # detection de visage + resize img + tête droite avec dlib
            frame = self.faceSearch.detectVisage(image)

            height, width, channels = frame.shape

            # traitement de frame pour prédiction avec VGG-16
            if(height == 224 and width == 224 and channels == 3):
                recognitionStr = self.predictP.predict(frame)

            # Affichage de la prédiciton si tête trouvé
            image = cv2.putText(image, 'Visage: '+recognitionStr, (5, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("Output", image)

            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break

        cv2.destroyAllWindows()
        self.cap.release()

    def image(self, img):
        recognitionStr = ""
        image = cv2.imread(img)

        # detection de visage + resize img + tête droite avec dlib
        frame = self.faceSearch.detectVisage(image)

        height, width, channels = frame.shape

        # traitement de frame pour prédiction avec VGG-16
        if(height == 224 and width == 224 and channels == 3):
            recognitionStr = self.predictP.predict(frame)

        # Affichage de la prédiciton si tête trouvé
        image = cv2.putText(image, 'Visage: '+recognitionStr, (5, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Output", image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cp = cameraPrediction()
    if(len(sys.argv) == 1):
        cp.camera()
    else:
        cp.image(sys.argv[1])
