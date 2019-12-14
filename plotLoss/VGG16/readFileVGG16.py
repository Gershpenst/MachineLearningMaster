#!/usr/bin/python3
# coding: utf-8

import subprocess
import math


def epochVar(strVal):
    return int(strVal[1].split("/")[0])

#time, 
def allVar(strVal):
    return int(strVal[2][:-1]), float(strVal[5]), float(strVal[8]), float(strVal[11]), float(strVal[14][:-2])


epoch = []
time = []
loss = []
accuracy = []
val_loss = []
val_accuracy = []


with open("VGG16File.txt", "r") as f:
    lignes = f.readlines()
    for ligne in lignes:
        strVal = ligne.split(" ")
        if(strVal[0] == 'Epoch'):
            epoch.append(epochVar(strVal))
        elif(strVal[0]==''):
            t, l, a, vl, va = allVar(strVal)
            time.append(t)
            loss.append(l)
            accuracy.append(a)
            val_loss.append(vl)
            val_accuracy.append(va)

fichier = open("cleanFileVGG16.txt", "w")
print("epoch - time - loss - accuracy - val_loss - val_accuracy")
fichier.write("epoch - time - loss - accuracy - val_loss - val_accuracy\n")
for i in range(0, len(epoch)):
    print(epoch[i]," ",time[i]," ",loss[i]," ",accuracy[i]," ",val_loss[i]," ",val_accuracy[i])
    sss = str(epoch[i]) + " " + str(time[i]) + " " + str(loss[i]) + " " + str(accuracy[i]) + " " + str(val_loss[i]) + " " + str(val_accuracy[i]) + "\n"
    fichier.write(sss)

fichier.close()