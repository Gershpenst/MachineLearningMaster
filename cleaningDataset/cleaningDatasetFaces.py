import os

from faceImageSearch import *

# Enlève les fichiers 'DS_store' à partir du chemin mis en paramétre
def removeDS_Store(dirtyDatasetRepertories):
    if('.DS_Store' in dirtyDatasetRepertories):
        dirtyDatasetRepertories.remove('.DS_Store')

# Crée un dossier où le chemin sera 'dir'
def createDir(dir):
    try: 
        os.makedirs(dir)
    except OSError:
        if not os.path.isdir(dir):
            raise

# Création du répertoire de nettoyage de la dataset si elle n'existe pas.
cleanDir = "datasetCleaned/"
createDir(cleanDir)

# Enregistre dans la variable 'dirtyDatasetRepertories' les sous-dossiers de 'dirtyRepertory' en enlevant le fichier 'DS_Store'
dirtyRepertory = './downloads/'
# dirtyRepertory = './Dossier/'
dirtyDatasetRepertories = os.listdir(dirtyRepertory)
removeDS_Store(dirtyDatasetRepertories)


# Détecte les visages dans une image
faceCleaning = faceImageSearch()

totalImg = []

# Parcours le dossier contenant tout les sous dossiers
for dirtyDir in dirtyDatasetRepertories:

    # Enregistre dans 'listDirDirtyData' tout les fichiers qui se trouve dans les sous-dossiers et enlève le fichier 'DS_store'.
    pathTotal = dirtyRepertory+dirtyDir+'/'
    listDirDirtyData = os.listdir(pathTotal)
    removeDS_Store(listDirDirtyData)

    # Création de chaques dossiers parcouru dans le dossier 'sale' dans le dossier 'propre'
    rep = cleanDir + dirtyDir
    createDir(rep)

    incr = 0
    print("-", dirtyDir)
    # Parcours les sous fichiers afin de les nettoyer (detection de visage)
    for (i, element) in enumerate(listDirDirtyData):
        # Detecte les visages et les enregistre dans les dossiers spéciphiques.
        if not element.endswith('.gif'): 
            incr = faceCleaning.detectVisage(pathTotal + element, path='./'+rep, dirname=dirtyDir, incr=incr)
    
    totalImg.append(element+" --> "+str(incr))

print("\n\n\n")
for total in totalImg:
    print(total)