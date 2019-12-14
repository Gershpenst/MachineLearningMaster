# Reconnaissance Facial à partir de Tiny Yolo et VGG-16 via Darkflow.
Utilisation: ```$ python3 cameraPrediction.py```

# Détails fichiers
Le fichier `cameraPredictionYolo.py` permet de répérer et de prédire la tête d'une personne.
Le fichier `faceImageSearchYolo` permet de chercher un visage à partir de `tiny Yolo` et de redimensionner les ROIs (224*224) pour le traitement dans `VGG-16`.
Le fichier `predictFaces.py` prédit le visage suivant les labels/célébrités choisit à partir de `VGG-16`. Celle-ci retournera la célébrité ayant la plus grande accuracy.

# Fichier manquant à télécharger
Mes poids pour `VGG-16`: [recognize_4class_1_0.5dropoutPlus.h5](https://google.com)
Les poids pour `tiny Yolo`: [tinyYolo](https://google.com)


