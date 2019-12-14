# MachineLearningMaster --> reconnaissanceFacialDlib
Cours de Master 2: Informatique - Machine Learning Keras

Utilisation: python3 cameraPrediction.py

Le fichier :  
    - "cameraPrediction.py" permet de répérer et de prédire la tête d'une personne.
    - "faceImageSearchDlib" permet de chercher un visage, puis retourner la tête si elle est penché et de redimensionner la ROI (224*224) pour le traitement dans VGG-16.
    - "predictFaces.py" prédit le visage suivant les labels/célébrités choisit à partir de VGG-16. Celle-ci retournera la célébrité ayant la plus grande accuracy

# Fichier manquant :
    - Mes poids pour VGG-16 "recognize_4class_1_0.5dropoutPlus.h5"
    - Les poids pour Dlib "shape_predictor_68_face_landmarks.dat"
