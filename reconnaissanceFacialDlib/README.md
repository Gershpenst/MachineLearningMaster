# Reconnaissance Facial à partir de dlib et VGG-16.
Utilisation:

* Camera: ```$ python3 cameraPrediction.py```
* Image: ```$ python3 cameraPrediction.py <path image>```

# Détails fichiers
Le fichier `cameraPrediction.py` permet de répérer et de prédire la tête d'une personne.

Le fichier `faceImageSearchDlib` permet de chercher un visage, puis retourner la tête si elle est penché et de redimensionner la ROI (224*224) pour le traitement dans `VGG-16`.

Le fichier `predictFaces.py` prédit le visage suivant les labels/célébrités choisit à partir de `VGG-16`. Celle-ci retournera la célébrité ayant la plus grande accuracy.

# Fichier manquant à télécharger
Mes poids pour `VGG-16`: [recognize_vgg16.h5](https://drive.google.com/open?id=1Lt90so56bYPsewnCBu8T8RhScuun0KhX).

Les poids pour `dlib`: [shape_predictor_68_face_landmarks.dat](https://drive.google.com/open?id=1Re7dLu17IC6PAd7sOzNhEGj7vIAQ_p4m).
