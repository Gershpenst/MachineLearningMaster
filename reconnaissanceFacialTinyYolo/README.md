# Reconnaissance Facial à partir de Tiny Yolo et VGG-16 via Darkflow.
Utilisation: ```$ python3 cameraPrediction.py```

# Détails fichiers
Le fichier `cameraPredictionYolo.py` permet de répérer et de prédire la tête d'une personne.

Le fichier `faceImageSearchYolo` permet de chercher un visage à partir de `tiny Yolo` et de redimensionner les ROIs (224*224) pour le traitement dans `VGG-16`.

Le fichier `predictFaces.py` prédit le visage suivant les labels/célébrités choisit à partir de `VGG-16`. Celle-ci retournera la célébrité ayant la plus grande accuracy.

# Fichier manquant à télécharger
Mes poids pour `VGG-16`: [recognize_vgg16.h5](https://drive.google.com/open?id=1Lt90so56bYPsewnCBu8T8RhScuun0KhX).

Les poids pour : 

* `tiny Yolo`: [tinyYolo basics](https://drive.google.com/open?id=1iOCNDxTVvvBY-3Q6BV2RNam6fB_mdIVP).
* `tiny Yolo`: [tinyYolo amélioré](https://drive.google.com/open?id=1pBftHxL_dfa67Rl2TI9U4QnlWaNfH-Qr).

Le fichier `ckpt`doit être mis à la racine dans le fichier. 

/!\ Dans le fichier `faceImageSearchYolo.py`, le paramétré `"load"` doit avoir le même poids que le fichier.