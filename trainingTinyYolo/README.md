# Entraînement de tiny Yolo pour la reconnaissance de visage.

L'apprentissage de `tiny Yolo` s'est fait à partir de [Darkflow](https://github.com/thtrieu/darkflow).

Utilisation: ```$ python3 flow --model cfg/tiny-yolo-1c.cfg --train adam --labels labels.txt --load bin/tiny-yolo.weights --annotation ~/image/annotation --dataset ~/image/Faces --gpu 1.0 --epoch 500```

# Détails fichiers
`--model` permet de spéciphier le modéle choisit. Ici, j'ai créé un fichier `tiny-yolo-1c.cfg` (voir `cfg/tiny-yolo-1c.cfg`) comportant les mêmes attributs que son fichier parent `tiny-yolo.cfg`. J'ai dû changer: 
* Dans la dernière couche de `convolutional`, le paramétre `filter` doit être mis à 30 vu que je posséde qu'une classe à lui faire apprendre ```5 * (lenClass + 5) -> 5 * (1 + 5) --> 30```.
* Dans la dernière couche `region`, j'ai dû changer le paramétre `classes` a été mit à 1.

`--train` permet de spéciphier l'optimisateur choisit pour la compiliation du modéle.

`--labels` permet de spéciphier dans un fichier txt, le(s) label(s) (ici, un label: `human_face`).

`--load` permet de charger/utiliser des poids.

`--annotation` contient les fichiers XML montrant les ROIs des images.

`--dataset` contient les images.

`--gpu` permet d'utiliser un pourcentage du GPU.

`--epoch` spéciphie l'epoch.

À la fin ou à chaques 125 epochs, les fichiers `data-00000-of-00001`, `index`, `meta` et `profil` sont créés.

# Labels pour tiny Yolo via Darkflow
Les labels ont été créer à la main à partir de [labelImg](https://github.com/tzutalin/labelImg) permettant de selectionner toutes les ROIs se trouvant dans une image. Un fichier XML (par image) est sauvegardé lorsqu'une ou plusieurs ROI(s) ont été selectionnées. Le(s) fichier(s) XML comprennent les coordonnées haut-gauche et bas-droit du rectangle, le nom de l'image référent, la longueur et largeur total de l'image... Un set de 1000 images ont été créé.

Certains fichiers ne pouvaient pas être lu, notamment les fichiers GIF. De plus, rarement, l'outil [labelImg](https://github.com/tzutalin/labelImg) sauvegarde le fichier XML en mettant la longueur et la hauteur de l'image à 0. Le fichier `fileSearchjpgRm.py` a permis de pallier à certains de mes problèmes.

Mon dataset avec [les images et les annotations](https://drive.google.com/open?id=1cbvnhoXK9ggQO2k8oPwfCbvpnMMRANlr).

Les poids utilisés: [tiny_yolo.weights](https://drive.google.com/open?id=1RDzBxnmB0kNIA89UIfMZiYLFj02OQTj5).

Les poids obtenus :
* `tiny Yolo`: [tinyYolo basics](https://drive.google.com/open?id=1iOCNDxTVvvBY-3Q6BV2RNam6fB_mdIVP). (bas des sourcils + moitié menton)
* `tiny Yolo`: [tinyYolo amélioré](https://drive.google.com/open?id=1pBftHxL_dfa67Rl2TI9U4QnlWaNfH-Qr). (front + menton)
