from imutils import face_utils
import dlib
import cv2

from faceImageSearchDlib import faceImageSearchDlib
from predictFaces import PredictionPeople

cap = cv2.VideoCapture(0)

faceSearch = faceImageSearchDlib()

listClass = ['ali',
'other', "Bill Gates",
"Brad Pitt", "Donald Trump",
"jacques chirac", "jean lassalle",
"Jean pierre coffe",
"Jennifer lopez", "Marine lepen",
"Tom cruise"]

# Initialisation du modéle pré-entrainer VGG-16
predictP = PredictionPeople("./recognize_vgg16.h5", listClass)

while True:
    recognitionStr = ""

    # Webcam
    _, image = cap.read()

    # detection de visage + resize img + tête droite avec dlib
    frame = faceSearch.detectVisage(image)

    height, width, channels = frame.shape

    # traitement de frame pour prédiction avec VGG-16
    if(height == 224 and width == 224 and channels == 3):
        recognitionStr = predictP.predict(frame)

    # Affichage de la prédiciton si tête trouvé
    image = cv2.putText(image, 'Visage: '+recognitionStr, (5, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Output", image)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
