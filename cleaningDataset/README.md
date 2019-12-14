# Nettoyage de donnée pour VGG-16.
Cette section permet de nettoyer les images, d'appliquer une rotation suivant la position de la tête et de redimensionner la ROI (Region Of Interest) pour satisfaire l'entrée de `VGG-16` (224*224)
# Détails fichiers
Le fichier `faceImageSearch.py` permet de repérer l'image et de la nettoyer
le fichier `cleaningDatasetFaces.py` permet de créer les dossiers nettoyés en parcourant le dossier `downloads` et ses sous fichiers.
# Fichier manquant à télécharger
Le fichier [shape_predictor_68_face_landmarks.dat](https://drive.google.com/open?id=1Re7dLu17IC6PAd7sOzNhEGj7vIAQ_p4m) permet de détecter le visage grâce à dlib. Ce fichier se met dans le dossier actuel.