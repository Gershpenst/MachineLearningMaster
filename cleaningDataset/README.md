# MachineLearningMaster --> cleaning dataset
Cours de Master 2: Informatique - Machine Learning Keras

Cette section permet de nettoyer les images, d'appliquer une rotation suivant la position de la tête et de redimensionner la ROI (Region Of Interest) pour satisfaire l'entrée de VGG-16 (224*224)

faceImageSearch.py permet de repérer l'image et de la nettoyer
cleaningDatasetFaces.py permet de créer les dossiers nettoyés en parcourant le dossier "downloads" et ses sous fichiers.

NB: Le fichier "shape_predictor_68_face_landmarks.dat" est très important, celle-ci permet de détecter le visage grâce à dlib.
