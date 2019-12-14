# MachineLearningMaster --> reconnaissanceFacialDlib
Cours de Master 2: Informatique - Machine Learning Keras

Utilisation: python3 cameraPrediction.py

Le fichier :  
    - "cameraPredictionYolo.py" permet de répérer et de prédire la tête d'une personne.
    - "faceImageSearchYolo" permet de chercher un visage (tiny Yolo) et de redimensionner les ROIs (224*224) pour le traitement dans VGG-16.
    - "predictFaces.py" prédit le visage suivant les labels/célébrités choisit à partir de VGG-16. Celle-ci retournera la célébrité ayant la plus grande accuracy

# Fichier manquant :
    - Mes poids pour VGG-16 "recognize_4class_1_0.5dropoutPlus.h5"
    - Mes poids pour tiny Yolo
