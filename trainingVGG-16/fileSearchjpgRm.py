import cv2
import os
from lxml import etree

elementImage = '/home/gespenst/image/Faces/'
elementXml = '/home/gespenst/image/annotation/'

countRm = 0

def rmFic(imgPath, countRm):
    doc, ext = os.path.splitext(imgPath)
    #print("doc: ", doc," et ext: ", ext)

    imgRm = elementImage+doc+ext
    imgXml = elementXml+doc+'.xml'

    os.remove(imgRm)
    if os. path. exists(imgXml) :
        os.remove(imgXml)
        countRm+=1
    return countRm

for imgFile in os.listdir(elementImage):
    #print(elementImage+imgFile)
    img = cv2.imread(elementImage+imgFile)

    if(img is None):
        countRm = rmFic(imgFile, countRm)
        #print("It is none")
        #exit(1)
    else:
        w, h, _ = img.shape

        doc, ext = os.path.splitext(imgFile)
        xmlPath = elementXml+doc+".xml"
        if os. path. exists(xmlPath) :
            tree = etree.parse(xmlPath)
            if(tree.xpath("/annotation/size/width")[0].text == '0' or tree.xpath("/annotation/size/height")[0].text == '0'):
                tree.xpath("/annotation/size/width")[0].text = str(w)
                tree.xpath("/annotation/size/height")[0].text = str(h)
                tree.write(xmlPath)
                print("fichier xml: ", xmlPath)


        #if(w == h):

            #if os. path. exists(elementXml+doc+".xml") :
        #if(imgFile == '34.NQJWSST4.jpg'):
        #    print("w: ", w, " and ", h, " and name: ",imgFile)

        if(w == 0 or h == 0):
            countRm = rmFic(imgFile, countRm)
            #print("w = ", w," et h = ",h)
            #exit(1)



print("Fichier supprimé: ", countRm)
