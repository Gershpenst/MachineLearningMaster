#!/usr/bin/python3
# coding: utf-8

import subprocess
import math

# subprocess.getoutput('make')

def moyenne(tab):
    i = 0
    moy = 0.0

    for i in range(0, len(tab)):
        moy += tab[i]

    return moy/len(tab)



def variance(tab, moy):
    i = 0
    var = 0.0

    for i in range(0, len(tab)):
        var += (moy - tab[i]) ** 2

    return var/len(tab)



def ecartType(var):
    return math.sqrt(var)


def bandeTest(tab):
    m = moyenne(tab)
    var = variance(tab, m)
    e = ecartType(var)
    return m, var, e

def epochVar(strVal):
    return int(strVal[5:-1])

def allVar(strVal):
    return int(strVal[0]), float(strVal[1]), float(strVal[2][:-2])

epoch = []

loss = []
mloss = []
eloss = []

val_loss = []
mval_loss = []
eval_loss = []

firstEpoch = True

with open("tinyYoloFile.txt", "r") as f:
    lignes = f.readlines()
    for ligne in lignes:
        strVal = ligne.split(" ")
        if(strVal[0][0:5] == 'epoch'):
            e = epochVar(strVal[0])
            epoch.append(e)
            if(not(firstEpoch)):
                ml, _, el = bandeTest(loss)
                mvl, _, evl = bandeTest(val_loss)

                mloss.append(ml)
                eloss.append(el)

                mval_loss.append(mvl)
                eval_loss.append(evl)

                loss.clear()
                val_loss.clear()
            else:
                firstEpoch = False
        
        elif(strVal[0] == 'END'):
            ml, _, el = bandeTest(loss)
            mvl, _, evl = bandeTest(val_loss)

            mloss.append(ml)
            eloss.append(el)

            mval_loss.append(mvl)
            eval_loss.append(evl)

            loss.clear()
            val_loss.clear()
        
        else:
            _, l, vl = allVar(strVal)
            loss.append(l)
            val_loss.append(vl)

fichier = open("cleanFileTinyYolo.txt", "w")
# print("epoch - loss - val_loss")
fichier.write("epoch - moy loss - ecart loss - moy val_loss - ecart val_loss \n")
for i in range(0, len(epoch)):
    #print(epoch[i]," ",mloss[i]," ",mval_loss[i])
    sss = str(epoch[i]) + " " + str(mloss[i]) + " " + str(eloss[i]) + " " + str(mval_loss[i]) + " " + str(eval_loss[i]) + "\n"
    fichier.write(sss)

fichier.close()