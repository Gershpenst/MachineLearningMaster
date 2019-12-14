import keras
from keras import backend as K
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Activation
from keras.layers.core import Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.optimizers import Adam, SGD
from keras.metrics import categorical_crossentropy

from keras.models import Model

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#----------------------------- Préparation d'images -----------------------------#

train_and_validaded_path = 'people/training_set'
valid_path = 'people/valid_set'

listClass = ['ali', 'other', "Bill Gates", "Brad Pitt", "Donald Trump", "jacques chirac", "jean lassalle", "Jean pierre coffe", "Jennifer lopez", "Marine lepen", "Tom cruise"]
lenClass = len(listClass)

train_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(train_and_validaded_path, target_size=(224, 224), classes=listClass, batch_size=15) #5
valid_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(valid_path, target_size=(224, 224), classes=listClass, batch_size=10)


#----------------------------- Entrainement et validation -----------------------------#


# VGG-16 architecture =>
# http://penseeartificielle.fr/focus-reseau-neurones-convolutifs/vgg-16-architecture/
# https://www.quora.com/What-is-the-VGG-neural-network
# https://github.com/rcmalli/keras-vggface/blob/master/keras_vggface/models.py
# Votre cours

model = Sequential()

model.add(Conv2D(64, (3,3), activation="relu", input_shape=(224, 224, 3)))   #Activation  MaxPooling2D Dropout
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(128, (3,3), activation="relu"))
model.add(Conv2D(128, (3,3), activation="relu"))
# model.add(Conv2D(128, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(256, (3,3), activation="relu"))
model.add(Conv2D(256, (3,3), activation="relu"))
model.add(Conv2D(256, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


model.add(Conv2D(512, (3,3), activation="relu"))
model.add(Conv2D(512, (3,3), activation="relu"))
model.add(Conv2D(512, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


model.add(Conv2D(512, (3,3), activation="relu"))
model.add(Conv2D(512, (3,3), activation="relu"))
model.add(Conv2D(512, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())

model.add(Dense(4096, activation='relu'))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(lenClass, activation='softmax')) 

model.summary()

print("Training and validation: \n\n")
model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy']) 
history = model.fit_generator(train_batches, validation_data=valid_batches, validation_steps=1, epochs=38, verbose=2)

print("Accuracy", history.history)

# save model
model.save('recognize_4class_1_0.5dropoutPlus.h5')


# Tout les tests que j'ai fait pour VGG-16 -->
'''
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 222, 222, 64)      1792
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 220, 220, 64)      36928
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 110, 110, 64)      0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 108, 108, 128)     73856
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 106, 106, 128)     147584
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 104, 104, 128)     147584
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 52, 52, 128)       0
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 50, 50, 256)       295168
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 48, 48, 256)       590080
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 46, 46, 256)       590080
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 23, 23, 256)       0
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 21, 21, 512)       1180160
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 19, 19, 512)       2359808
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 17, 17, 512)       2359808
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 8, 8, 512)         0
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 6, 6, 512)         2359808
_________________________________________________________________
conv2d_13 (Conv2D)           (None, 4, 4, 512)         2359808
_________________________________________________________________
conv2d_14 (Conv2D)           (None, 2, 2, 512)         2359808
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 1, 1, 512)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 512)               0
_________________________________________________________________
dense_1 (Dense)              (None, 4096)              2101248
_________________________________________________________________
dense_2 (Dense)              (None, 4096)              16781312
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 8194
=================================================================
Total params: 33,753,026
Trainable params: 33,753,026
Non-trainable params: 0
_________________________________________________________________
Training and validation:


WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/keras/optimizers.py:793: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.

WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:3576: The name tf.log is deprecated. Please use tf.math.log instead.

WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/tensorflow/python/ops/math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:1033: The name tf.assign_add is deprecated. Please use tf.compat.v1.assign_add instead.

Epoch 1/5
2019-11-12 11:42:01.752922: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
 - 23s - loss: 0.7425 - acc: 0.4667 - val_loss: 0.6986 - val_acc: 0.5000
Epoch 2/5
 - 20s - loss: 0.6902 - acc: 0.4667 - val_loss: 0.6302 - val_acc: 1.0000
Epoch 3/5
 - 22s - loss: 0.6526 - acc: 0.7000 - val_loss: 0.4979 - val_acc: 1.0000
Epoch 4/5
 - 22s - loss: 0.5712 - acc: 0.7667 - val_loss: 1.0660 - val_acc: 0.5000
Epoch 5/5
 - 24s - loss: 0.6849 - acc: 0.5667 - val_loss: 0.6928 - val_acc: 0.5000
Prediction:


predictions avec  {'ali': 0, 'other': 1}  :
 [0.55341005 0.6894778  0.6882495  0.5662309  0.5792738  0.6299374 ]
 [Ali          PasAli     Ali        PasAli    PasAli     Ali      ]
'''



'''
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 222, 222, 64)      1792
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 220, 220, 64)      36928
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 110, 110, 64)      0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 108, 108, 128)     73856
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 106, 106, 128)     147584
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 53, 53, 128)       0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 51, 51, 256)       295168
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 49, 49, 256)       590080
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 47, 47, 256)       590080
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 23, 23, 256)       0
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 21, 21, 512)       1180160
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 19, 19, 512)       2359808
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 17, 17, 512)       2359808
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 8, 8, 512)         0
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 6, 6, 512)         2359808
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 4, 4, 512)         2359808
_________________________________________________________________
conv2d_13 (Conv2D)           (None, 2, 2, 512)         2359808
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 1, 1, 512)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 512)               0
_________________________________________________________________
dense_1 (Dense)              (None, 4096)              2101248
_________________________________________________________________
dense_2 (Dense)              (None, 4096)              16781312
_________________________________________________________________
dense_3 (Dense)              (None, 4)                 16388
=================================================================
Total params: 33,613,636
Trainable params: 33,613,636
Non-trainable params: 0
_________________________________________________________________
2019-11-13 22:08:51.406114: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-11-13 22:08:51.427920: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2793480000 Hz
2019-11-13 22:08:51.428275: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b72a85a170 executing computations on platform Host. Devices:
2019-11-13 22:08:51.428315: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): <undefined>, <undefined>
2019-11-13 22:08:51.509695: W tensorflow/compiler/jit/mark_for_compilation_pass.cc:1412] (One-time warning): Not using XLA:CPU for cluster because envvar TF_XLA_FLAGS=--tf_xla_cpu_global_jit was not set.  If you want XLA:CPU, either set that envvar, or use experimental_jit_scope to enable XLA:CPU.  To confirm that XLA is active, pass --vmodule=xla_compilation_cache=1 (as a proper command-line flag, not via TF_XLA_FLAGS) or set the envvar XLA_FLAGS=--xla_hlo_profile.
Training and validation:


WARNING:tensorflow:From /home/gespenst/.local/lib/python3.5/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/25
 - 392s - loss: 1.1081 - accuracy: 0.4689 - val_loss: 0.8121 - val_accuracy: 0.5000
Epoch 2/25
 - 391s - loss: 1.0054 - accuracy: 0.4852 - val_loss: 1.3372 - val_accuracy: 0.2000
Epoch 3/25
 - 410s - loss: 0.9555 - accuracy: 0.4557 - val_loss: 1.0342 - val_accuracy: 0.7000
Epoch 4/25
 - 402s - loss: 0.6615 - accuracy: 0.7967 - val_loss: 0.7065 - val_accuracy: 0.7000
Epoch 5/25
 - 398s - loss: 0.4510 - accuracy: 0.8721 - val_loss: 2.5124 - val_accuracy: 0.7000
Epoch 6/25
 - 398s - loss: 0.4328 - accuracy: 0.8754 - val_loss: 0.4278 - val_accuracy: 0.8000
Epoch 7/25
 - 401s - loss: 0.3390 - accuracy: 0.8852 - val_loss: 1.7832 - val_accuracy: 0.7000
Epoch 8/25
 - 399s - loss: 0.2897 - accuracy: 0.8820 - val_loss: 0.0048 - val_accuracy: 1.0000
Epoch 9/25
 - 399s - loss: 0.3155 - accuracy: 0.8852 - val_loss: 0.4815 - val_accuracy: 0.8000
Epoch 10/25
 - 398s - loss: 0.2535 - accuracy: 0.8902 - val_loss: 0.5917 - val_accuracy: 0.7000
Epoch 11/25
 - 397s - loss: 0.2078 - accuracy: 0.9131 - val_loss: 2.0858 - val_accuracy: 0.7000
Epoch 12/25
 - 396s - loss: 0.1915 - accuracy: 0.9164 - val_loss: 0.5988 - val_accuracy: 0.9000
Epoch 13/25
 - 394s - loss: 0.2644 - accuracy: 0.9131 - val_loss: 0.4902 - val_accuracy: 0.8000
Epoch 14/25
 - 394s - loss: 0.1732 - accuracy: 0.9311 - val_loss: 1.6458 - val_accuracy: 0.7000
Epoch 15/25
 - 394s - loss: 0.1251 - accuracy: 0.9492 - val_loss: 0.5297 - val_accuracy: 0.8000
Epoch 16/25
 - 391s - loss: 0.1081 - accuracy: 0.9525 - val_loss: 0.0057 - val_accuracy: 1.0000
Epoch 17/25
 - 391s - loss: 0.1769 - accuracy: 0.9459 - val_loss: 0.0093 - val_accuracy: 1.0000
Epoch 18/25
 - 391s - loss: 0.1722 - accuracy: 0.9607 - val_loss: 0.3869 - val_accuracy: 0.9000
Epoch 19/25
 - 391s - loss: 0.1408 - accuracy: 0.9525 - val_loss: 0.4015 - val_accuracy: 0.9000
Epoch 20/25
 - 389s - loss: 0.1394 - accuracy: 0.9590 - val_loss: 1.7835 - val_accuracy: 0.6000
Epoch 21/25
 - 392s - loss: 0.1139 - accuracy: 0.9492 - val_loss: 1.9401 - val_accuracy: 0.8000
Epoch 22/25
 - 398s - loss: 0.0599 - accuracy: 0.9852 - val_loss: 0.5540 - val_accuracy: 0.9000
Epoch 23/25
 - 398s - loss: 0.0613 - accuracy: 0.9754 - val_loss: 0.4681 - val_accuracy: 0.7000
Epoch 24/25
 - 397s - loss: 0.0167 - accuracy: 0.9967 - val_loss: 3.4862 - val_accuracy: 0.0000e+00
Epoch 25/25
 - 397s - loss: 0.0182 - accuracy: 0.9918 - val_loss: 5.9655 - val_accuracy: 0.6000
Accuracy {'val_accuracy': [0.5, 0.20000000298023224, 0.699999988079071, 0.699999988079071, 0.699999988079071, 0.800000011920929, 0.699999988079071, 1.0, 0.800000011920929, 0.699999988079071, 0.699999988079071, 0.8999999761581421, 0.800000011920929, 0.699999988079071, 0.800000011920929, 1.0, 1.0, 0.8999999761581421, 0.8999999761581421, 0.6000000238418579, 0.800000011920929, 0.8999999761581421, 0.699999988079071, 0.0, 0.6000000238418579], 'val_loss': [0.8121140599250793, 1.3371601104736328, 1.0341637134552002, 0.706468403339386, 2.512434959411621, 0.42778366804122925, 1.7832047939300537, 0.004764988087117672, 0.48153990507125854, 0.5916651487350464, 2.085817337036133, 0.5988141298294067, 0.49020734429359436, 1.645753264427185, 0.5296686887741089, 0.0057219755835831165, 0.009333522990345955, 0.38694196939468384, 0.40154752135276794, 1.7835102081298828, 1.9400783777236938, 0.5539690852165222, 0.4681081771850586, 3.4861879348754883, 5.965487957000732], 'loss': [1.10814768075943, 1.005366653692527, 0.9554877193247686, 0.6615095492757734, 0.4509863642761942, 0.4327767984788926, 0.33899305995217843, 0.2896775375349355, 0.31550051165042353, 0.25353497172682926, 0.207754697897815, 0.19147598650020195, 0.26442074464237103, 0.1732335671144118, 0.12513903685672242, 0.1080702939053149, 0.17690552340940724, 0.17215901950344903, 0.14077928153597358, 0.1393907628006837, 0.11388173027175906, 0.059949599978685014, 0.061313132020184076, 0.016679009289002008, 0.018186970890943982], 'accuracy': [0.46885246, 0.4852459, 0.4557377, 0.79672134, 0.87213117, 0.87540984, 0.8852459, 0.8819672, 0.8852459, 0.89016396, 0.9131147, 0.91639346, 0.9131147, 0.9311475, 0.9491803, 0.95245904, 0.94590163, 0.96065575, 0.95245904, 0.9590164, 0.9491803, 0.9852459, 0.97540987, 0.9967213, 0.9918033]}


 [0.]


Prediction:


20/20 [==============================] - 5s 232ms/step
predictions avec  {'Bill Gates': 2, 'other': 1, 'ali': 0, 'Brad Pitt': 3}  :
 [8.96538296e-15 8.97576690e-01 1.00000000e+00 1.60954065e-14
 2.69451040e-34 1.37080741e-21 1.04561350e-05 2.96336924e-22
 4.41894787e-21 1.37564570e-01 4.23626528e-33 2.53595468e-02
 9.06115460e-19 1.08849978e-08 3.76182184e-34 1.17393165e-05
 5.13646237e-10 1.84689252e-05 6.01934593e-31 8.87022118e-14]


[8.9653830e-15 5.6081065e-14 1.0000000e+00 5.0101173e-12]
[0.8975767  0.04957733 0.00551722 0.04732871]
[1.0000000e+00 4.6454060e-12 3.9690955e-11 2.0605943e-10]
[1.6095406e-14 7.7605994e-10 4.5611982e-12 1.0000000e+00]
[2.6945104e-34 5.0314511e-28 1.0000000e+00 7.0129557e-27]
[1.3708074e-21 2.0350572e-18 1.0000000e+00 3.3793233e-20]
[1.0456135e-05 2.2098719e-01 4.6778159e-06 7.7899772e-01]
[2.9633692e-22 3.6374917e-22 1.0000000e+00 1.1274966e-21]
[4.4189479e-21 7.4849377e-17 1.0000000e+00 7.6095935e-18]
[0.13756457 0.09508274 0.04627509 0.7210776 ]
[4.2362653e-33 1.0131969e-31 1.0000000e+00 3.8067536e-30]
[0.02535955 0.00266149 0.0992702  0.87270874]
[9.0611546e-19 1.1545044e-11 9.6475877e-16 1.0000000e+00]
[1.0884998e-08 1.5453190e-07 9.9999988e-01 1.2938612e-08]
[3.7618218e-34 3.5571215e-37 1.0000000e+00 0.0000000e+00]
[1.17393165e-05 1.63945579e-03 4.03627346e-05 9.98308420e-01]
[5.1364624e-10 2.6712819e-09 1.0000000e+00 9.9866169e-09]
[1.8468925e-05 5.1733691e-02 6.7457149e-06 9.4824106e-01]
[6.0193459e-31 6.9633175e-20 5.1823471e-26 1.0000000e+00]
[8.8702212e-14 8.3628105e-15 1.0000000e+00 1.0087359e-14]
'''



# Avec 2 Dense de 0.5
'''
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 222, 222, 64)      1792
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 220, 220, 64)      36928
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 110, 110, 64)      0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 108, 108, 128)     73856
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 106, 106, 128)     147584
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 53, 53, 128)       0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 51, 51, 256)       295168
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 49, 49, 256)       590080
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 47, 47, 256)       590080
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 23, 23, 256)       0
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 21, 21, 512)       1180160
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 19, 19, 512)       2359808
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 17, 17, 512)       2359808
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 8, 8, 512)         0
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 6, 6, 512)         2359808
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 4, 4, 512)         2359808
_________________________________________________________________
conv2d_13 (Conv2D)           (None, 2, 2, 512)         2359808
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 1, 1, 512)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 512)               0
_________________________________________________________________
dense_1 (Dense)              (None, 4096)              2101248
_________________________________________________________________
dropout_1 (Dropout)          (None, 4096)              0
_________________________________________________________________
dense_2 (Dense)              (None, 4096)              16781312
_________________________________________________________________
dropout_2 (Dropout)          (None, 4096)              0
_________________________________________________________________
dense_3 (Dense)              (None, 4)                 16388
=================================================================
Total params: 33,613,636
Trainable params: 33,613,636
Non-trainable params: 0
_________________________________________________________________
2019-11-14 10:06:44.241484: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-11-14 10:06:44.397905: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2793480000 Hz
2019-11-14 10:06:44.398110: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x555942b8f850 executing computations on platform Host. Devices:
2019-11-14 10:06:44.398129: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): <undefined>, <undefined>
2019-11-14 10:06:44.459212: W tensorflow/compiler/jit/mark_for_compilation_pass.cc:1412] (One-time warning): Not using XLA:CPU for cluster because envvar TF_XLA_FLAGS=--tf_xla_cpu_global_jit was not set.  If you want XLA:CPU, either set that envvar, or use experimental_jit_scope to enable XLA:CPU.  To confirm that XLA is active, pass --vmodule=xla_compilation_cache=1 (as a proper command-line flag, not via TF_XLA_FLAGS) or set the envvar XLA_FLAGS=--xla_hlo_profile.
Training and validation:


WARNING:tensorflow:From /home/gespenst/.local/lib/python3.5/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/25
 - 412s - loss: 1.0990 - accuracy: 0.4443 - val_loss: 0.9987 - val_accuracy: 0.6000
Epoch 2/25
 - 410s - loss: 1.0330 - accuracy: 0.4541 - val_loss: 1.1961 - val_accuracy: 0.5000
Epoch 3/25
 - 408s - loss: 1.0509 - accuracy: 0.4689 - val_loss: 1.3776 - val_accuracy: 0.2000
Epoch 4/25
 - 406s - loss: 0.9883 - accuracy: 0.4803 - val_loss: 1.2031 - val_accuracy: 0.5000
Epoch 5/25
 - 407s - loss: 0.9823 - accuracy: 0.4656 - val_loss: 1.2905 - val_accuracy: 0.3000
Epoch 6/25
 - 406s - loss: 0.9447 - accuracy: 0.5934 - val_loss: 1.5772 - val_accuracy: 0.6000
Epoch 7/25
 - 405s - loss: 0.6072 - accuracy: 0.8033 - val_loss: 0.9280 - val_accuracy: 0.7000
Epoch 8/25
 - 406s - loss: 0.5307 - accuracy: 0.8459 - val_loss: 0.1263 - val_accuracy: 1.0000
Epoch 9/25
 - 406s - loss: 0.4529 - accuracy: 0.8623 - val_loss: 0.3523 - val_accuracy: 0.9000
Epoch 10/25
 - 405s - loss: 0.4012 - accuracy: 0.8787 - val_loss: 1.3181 - val_accuracy: 0.6000
Epoch 11/25
 - 400s - loss: 0.3634 - accuracy: 0.8820 - val_loss: 0.4910 - val_accuracy: 0.8000
Epoch 12/25
 - 400s - loss: 0.3202 - accuracy: 0.8852 - val_loss: 1.4753 - val_accuracy: 0.5000
Epoch 13/25
 - 403s - loss: 0.2978 - accuracy: 0.8984 - val_loss: 0.5584 - val_accuracy: 0.8000
Epoch 14/25
 - 401s - loss: 0.2437 - accuracy: 0.9213 - val_loss: 0.6468 - val_accuracy: 0.9000
Epoch 15/25
 - 402s - loss: 0.2032 - accuracy: 0.9246 - val_loss: 0.5340 - val_accuracy: 0.9000
Epoch 16/25
 - 398s - loss: 0.2264 - accuracy: 0.9164 - val_loss: 0.1174 - val_accuracy: 1.0000
Epoch 17/25
 - 399s - loss: 0.1977 - accuracy: 0.9459 - val_loss: 0.3424 - val_accuracy: 0.8000
Epoch 18/25
 - 398s - loss: 0.1518 - accuracy: 0.9475 - val_loss: 0.6944 - val_accuracy: 0.7000
Epoch 19/25
 - 400s - loss: 0.1274 - accuracy: 0.9607 - val_loss: 1.0151 - val_accuracy: 0.7000
Epoch 20/25
 - 401s - loss: 0.1879 - accuracy: 0.9410 - val_loss: 0.5499 - val_accuracy: 0.9000
Epoch 21/25
 - 405s - loss: 0.0927 - accuracy: 0.9623 - val_loss: 1.2736 - val_accuracy: 0.7000
Epoch 22/25
 - 410s - loss: 0.0451 - accuracy: 0.9852 - val_loss: 0.0192 - val_accuracy: 1.0000
Epoch 23/25
 - 411s - loss: 0.0221 - accuracy: 0.9934 - val_loss: 1.5771 - val_accuracy: 0.8000
Epoch 24/25
 - 410s - loss: 0.0093 - accuracy: 0.9984 - val_loss: 1.9582 - val_accuracy: 0.0000e+00
Epoch 25/25
 - 408s - loss: 0.0063 - accuracy: 0.9984 - val_loss: 3.0352 - val_accuracy: 0.7000
Accuracy {'val_loss': [0.998659610748291, 1.1961158514022827, 1.377631425857544, 1.203126311302185, 1.2905434370040894, 1.5772075653076172, 0.9279965162277222, 0.12634849548339844, 0.35229334235191345, 1.31808602809906, 0.4910299777984619, 1.4752753973007202, 0.5583982467651367, 0.6468216776847839, 0.5340191125869751, 0.1173982098698616, 0.342416375875473, 0.6943538188934326, 1.0150947570800781, 0.5498597025871277, 1.2736256122589111, 0.01924307458102703, 1.5771288871765137, 1.9582444429397583, 3.035158634185791], 'loss': [1.099015037544438, 1.032994345563357, 1.0509460558656787, 0.9882993121616176, 0.9823303017459932, 0.9447360830228837, 0.6071830206230039, 0.5306736715748662, 0.4529329059553928, 0.4011924697971735, 0.3634127147007184, 0.32020187371822656, 0.2977546646000176, 0.2436701464199568, 0.2031572354675011, 0.2264155745277273, 0.1976961365702455, 0.15178854232148237, 0.12740396812642146, 0.18787245310287276, 0.09267854552332419, 0.04513813960462268, 0.022084662788815466, 0.00927989252900078, 0.006292175124754841], 'val_accuracy': [0.6000000238418579, 0.5, 0.20000000298023224, 0.5, 0.30000001192092896, 0.6000000238418579, 0.699999988079071, 1.0, 0.8999999761581421, 0.6000000238418579, 0.800000011920929, 0.5, 0.800000011920929, 0.8999999761581421, 0.8999999761581421, 1.0, 0.800000011920929, 0.699999988079071, 0.699999988079071, 0.8999999761581421, 0.699999988079071, 1.0, 0.800000011920929, 0.0, 0.699999988079071], 'accuracy': [0.4442623, 0.45409837, 0.46885246, 0.48032787, 0.46557376, 0.5934426, 0.8032787, 0.84590167, 0.8622951, 0.8786885, 0.8819672, 0.8852459, 0.89836067, 0.9213115, 0.9245902, 0.91639346, 0.94590163, 0.947541, 0.96065575, 0.9409836, 0.96229506, 0.9852459, 0.9934426, 0.99836063, 0.99836063]}


 [1.]


Prediction:


20/20 [==============================] - 4s 176ms/step
predictions avec  {'ali': 0, 'other': 1, 'Brad Pitt': 3, 'Bill Gates': 2}  :
 [9.9998724e-01 5.5261282e-16 5.2809794e-16 8.7063804e-35 1.5657744e-28
 1.1055375e-20 2.7795630e-10 2.9211859e-07 1.8375219e-10 3.2795533e-20
 0.0000000e+00 1.2939717e-05 1.9223194e-18 3.4805172e-15 1.9019584e-12
 5.5874307e-07 3.1402384e-04 8.1644327e-14 6.4420658e-13 1.3302019e-17]


[9.99987245e-01 1.25739725e-05 2.39157913e-07 2.98623820e-10]
[5.5261282e-16 5.4809263e-24 1.0000000e+00 7.7860531e-22]
[5.2809794e-16 1.4915258e-22 1.0000000e+00 1.8867920e-19]
[8.7063804e-35 0.0000000e+00 1.0000000e+00 0.0000000e+00]
[1.5657744e-28 0.0000000e+00 1.0000000e+00 9.2495277e-35]
[1.1055375e-20 1.8448609e-30 1.0000000e+00 1.5802931e-24]
[2.7795630e-10 1.9411902e-06 8.0250612e-11 9.9999809e-01]
[2.9211859e-07 1.1213529e-09 3.8619037e-03 9.9613780e-01]
[1.8375219e-10 2.0906288e-07 8.9865643e-10 9.9999976e-01]
[3.2795533e-20 8.7172410e-29 1.0000000e+00 3.6623688e-24]
[0. 0. 1. 0.]
[1.2939717e-05 3.4447150e-09 9.9978334e-01 2.0373998e-04]
[1.9223194e-18 2.3432297e-26 1.0000000e+00 1.5274330e-21]
[3.4805172e-15 5.5338151e-13 1.5534100e-15 1.0000000e+00]
[1.9019584e-12 1.9633567e-03 2.3596611e-11 9.9803668e-01]
[5.5874307e-07 9.9999642e-01 9.5184227e-08 2.8565564e-06]
[3.1402384e-04 9.8090553e-01 1.2675997e-05 1.8767780e-02]
[8.1644327e-14 3.2118056e-20 1.0000000e+00 2.2532673e-16]
[6.4420658e-13 2.7965974e-09 2.2548570e-13 1.0000000e+00]
[1.3302019e-17 2.6017972e-17 5.8684097e-17 1.0000000e+00]

'''

# Avec un Dense 0.5 --- recognize_4class_1_0.5dropout.h5
'''
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 222, 222, 64)      1792
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 220, 220, 64)      36928
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 110, 110, 64)      0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 108, 108, 128)     73856
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 106, 106, 128)     147584
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 53, 53, 128)       0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 51, 51, 256)       295168
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 49, 49, 256)       590080
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 47, 47, 256)       590080
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 23, 23, 256)       0
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 21, 21, 512)       1180160
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 19, 19, 512)       2359808
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 17, 17, 512)       2359808
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 8, 8, 512)         0
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 6, 6, 512)         2359808
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 4, 4, 512)         2359808
_________________________________________________________________
conv2d_13 (Conv2D)           (None, 2, 2, 512)         2359808
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 1, 1, 512)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 512)               0
_________________________________________________________________
dense_1 (Dense)              (None, 4096)              2101248
_________________________________________________________________
dense_2 (Dense)              (None, 4096)              16781312
_________________________________________________________________
dropout_1 (Dropout)          (None, 4096)              0
_________________________________________________________________
dense_3 (Dense)              (None, 4)                 16388
=================================================================
Total params: 33,613,636
Trainable params: 33,613,636
Non-trainable params: 0
_________________________________________________________________
Training and validation:


2019-11-15 10:24:41.131033: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-11-15 10:24:41.358814: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2793480000 Hz
2019-11-15 10:24:41.384442: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560b0a6205c0 executing computations on platform Host. Devices:
2019-11-15 10:24:41.384477: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): <undefined>, <undefined>
2019-11-15 10:24:41.657523: W tensorflow/compiler/jit/mark_for_compilation_pass.cc:1412] (One-time warning): Not using XLA:CPU for cluster because envvar TF_XLA_FLAGS=--tf_xla_cpu_global_jit was not set.  If you want XLA:CPU, either set that envvar, or use experimental_jit_scope to enable XLA:CPU.  To confirm that XLA is active, pass --vmodule=xla_compilation_cache=1 (as a proper command-line flag, not via TF_XLA_FLAGS) or set the envvar XLA_FLAGS=--xla_hlo_profile.
WARNING:tensorflow:From /home/gespenst/.local/lib/python3.5/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/25
 - 390s - loss: 1.0831 - accuracy: 0.4393 - val_loss: 1.1715 - val_accuracy: 0.3000
Epoch 2/25
 - 390s - loss: 1.0036 - accuracy: 0.4525 - val_loss: 0.9696 - val_accuracy: 0.3000
Epoch 3/25
 - 389s - loss: 0.9736 - accuracy: 0.4787 - val_loss: 1.5559 - val_accuracy: 0.2000
Epoch 4/25
 - 390s - loss: 0.9367 - accuracy: 0.5984 - val_loss: 0.9019 - val_accuracy: 0.7000
Epoch 5/25
 - 389s - loss: 0.6071 - accuracy: 0.8377 - val_loss: 1.4978 - val_accuracy: 0.6000
Epoch 6/25
 - 388s - loss: 0.4842 - accuracy: 0.8623 - val_loss: 0.7117 - val_accuracy: 0.7000
Epoch 7/25
 - 387s - loss: 0.3982 - accuracy: 0.8770 - val_loss: 0.6132 - val_accuracy: 0.7000
Epoch 8/25
 - 387s - loss: 0.3507 - accuracy: 0.8787 - val_loss: 0.0789 - val_accuracy: 1.0000
Epoch 9/25
 - 388s - loss: 0.2964 - accuracy: 0.8902 - val_loss: 1.5853 - val_accuracy: 0.9000
Epoch 10/25
 - 392s - loss: 0.2445 - accuracy: 0.9016 - val_loss: 1.1635 - val_accuracy: 0.5000
Epoch 11/25
 - 396s - loss: 0.2377 - accuracy: 0.9115 - val_loss: 0.1724 - val_accuracy: 1.0000
Epoch 12/25
 - 398s - loss: 0.3043 - accuracy: 0.9016 - val_loss: 0.4358 - val_accuracy: 0.8000
Epoch 13/25
 - 398s - loss: 0.2149 - accuracy: 0.9213 - val_loss: 1.4150 - val_accuracy: 0.5000
Epoch 14/25
 - 401s - loss: 0.1929 - accuracy: 0.9148 - val_loss: 0.8205 - val_accuracy: 0.7000
Epoch 15/25
 - 400s - loss: 0.1518 - accuracy: 0.9443 - val_loss: 0.0671 - val_accuracy: 1.0000
Epoch 16/25
 - 399s - loss: 0.1013 - accuracy: 0.9639 - val_loss: 5.5074 - val_accuracy: 0.0000e+00
Epoch 17/25
 - 400s - loss: 0.1103 - accuracy: 0.9623 - val_loss: 0.4129 - val_accuracy: 0.7000
Epoch 18/25
 - 399s - loss: 0.2496 - accuracy: 0.9328 - val_loss: 0.4675 - val_accuracy: 0.9000
Epoch 19/25
 - 398s - loss: 0.1582 - accuracy: 0.9393 - val_loss: 1.3380 - val_accuracy: 0.7000
Epoch 20/25
 - 398s - loss: 0.0858 - accuracy: 0.9705 - val_loss: 1.8401 - val_accuracy: 0.8000
Epoch 21/25
 - 396s - loss: 0.0418 - accuracy: 0.9902 - val_loss: 0.5649 - val_accuracy: 0.9000
Epoch 22/25
 - 397s - loss: 0.0139 - accuracy: 0.9951 - val_loss: 1.0546 - val_accuracy: 0.9000
Epoch 23/25
 - 397s - loss: 0.0642 - accuracy: 0.9721 - val_loss: 1.5526 - val_accuracy: 0.8000
Epoch 24/25
 - 395s - loss: 0.1179 - accuracy: 0.9639 - val_loss: 0.0000e+00 - val_accuracy: 1.0000
Epoch 25/25
 - 396s - loss: 0.0150 - accuracy: 0.9967 - val_loss: 4.4476e-04 - val_accuracy: 1.0000
Accuracy {'loss': [1.083092079787958, 1.0035976103094757, 0.9736374860904256, 0.9366754811318194, 0.6071317998356507, 0.4842289188327115, 0.39815666025779284, 0.3507475759528699, 0.2963822136526225, 0.24447872294265716, 0.23769633601549403, 0.30427322856753636, 0.2148966402116949, 0.19290067337834577, 0.15178719642240213, 0.10133251057547966, 0.11032131195886599, 0.2495645122203668, 0.15824934770931637, 0.08582098585975846, 0.041815231018293576, 0.013884131215474967, 0.06418193476509364, 0.11793057008271418, 0.015021730804949724], 'val_loss': [1.1714929342269897, 0.9695857763290405, 1.5559403896331787, 0.9019119143486023, 1.4977731704711914, 0.7116969227790833, 0.6132414937019348, 0.07892070710659027, 1.5853389501571655, 1.1634573936462402, 0.17241665720939636, 0.4357932209968567, 1.4150272607803345, 0.8205406069755554, 0.06710690259933472, 5.507392883300781, 0.41293230652809143, 0.4674971103668213, 1.3379534482955933, 1.8401038646697998, 0.5649466514587402, 1.0546282529830933, 1.5526179075241089, 0.0, 0.00044475687900558114], 'val_accuracy': [0.30000001192092896, 0.30000001192092896, 0.20000000298023224, 0.699999988079071, 0.6000000238418579, 0.699999988079071, 0.699999988079071, 1.0, 0.8999999761581421, 0.5, 1.0, 0.800000011920929, 0.5, 0.699999988079071, 1.0, 0.0, 0.699999988079071, 0.8999999761581421, 0.699999988079071, 0.800000011920929, 0.8999999761581421, 0.8999999761581421, 0.800000011920929, 1.0, 1.0], 'accuracy': [0.43934426, 0.452459, 0.47868854, 0.59836066, 0.8377049, 0.8622951, 0.8770492, 0.8786885, 0.89016396, 0.90163934, 0.9114754, 0.90163934, 0.9213115, 0.9147541, 0.94426227, 0.9639344, 0.96229506, 0.9327869, 0.9393443, 0.9704918, 0.9901639, 0.99508196, 0.97213113, 0.9639344, 0.9967213]}


 [0.]


Prediction:


20/20 [==============================] - 3s 166ms/step
predictions avec  {'other': 1, 'Brad Pitt': 3, 'Bill Gates': 2, 'ali': 0}  :
 [9.05644086e-37 0.00000000e+00 7.73939197e-20 2.27795427e-20
 2.19659790e-10 1.03709137e-03 2.27422604e-23 3.56796386e-06
 1.00000000e+00 5.56658142e-16 1.02831524e-26 1.88989618e-27
 1.46143153e-04 1.64718498e-02 1.07826703e-12 5.80516389e-05
 3.22324388e-13 7.40416449e-19 9.66381311e-01 1.62483415e-21]


[9.0564409e-37 7.1311077e-29 1.0000000e+00 3.1182515e-29]
[0.000000e+00 2.312051e-37 1.000000e+00 6.504445e-38]
[7.739392e-20 5.942397e-14 1.991577e-15 1.000000e+00]
[2.2779543e-20 2.0913477e-14 8.3897105e-16 1.0000000e+00]
[2.1965979e-10 6.9295142e-10 1.0000000e+00 1.2560540e-09]
[0.00103709 0.82829237 0.00090338 0.16976722]
[2.2742260e-23 9.5724215e-19 1.0000000e+00 3.0377838e-18]
[3.5679639e-06 4.6609086e-05 9.9994731e-01 2.4685307e-06]
[1.0000000e+00 2.0651468e-10 1.7862447e-09 2.3003976e-08]
[5.5665814e-16 4.8133698e-12 1.0000000e+00 1.1879435e-12]
[1.02831524e-26 2.08273622e-19 2.21333337e-20 1.00000000e+00]
[1.8898962e-27 1.9516228e-26 1.0000000e+00 3.4654063e-27]
[1.4614315e-04 9.2136538e-01 7.1728427e-05 7.8416668e-02]
[0.01647185 0.04981456 0.03743047 0.89628315]
[1.0782670e-12 8.7483398e-10 1.5610601e-09 1.0000000e+00]
[5.8051639e-05 8.6308420e-01 7.5108430e-05 1.3678263e-01]
[3.2232439e-13 7.8324500e-11 1.1470341e-09 1.0000000e+00]
[7.4041645e-19 7.8030768e-18 1.0000000e+00 1.9747951e-17]
[0.9663813  0.00194425 0.0290463  0.00262826]
[1.6248341e-21 6.7848121e-18 1.0000000e+00 5.2525601e-20]
'''





# recognize_4class_1_0.5dropoutAll.h5
'''
Found 2239 images belonging to 11 classes.
Found 355 images belonging to 11 classes.
Found 63 images belonging to 11 classes.
predictions avec  {'Marine lepen': 9, 'jacques chirac': 5, 'other': 1, 'Brad Pitt': 3, 'jean lassalle': 6, 'Jennifer lopez': 8, 'Donald Trump': 4, 'Bill Gates': 2, 'ali': 0, 'Tom cruise': 10, 'Jean pierre coffe': 7}  :  <keras.preprocessing.image.DirectoryIterator object at 0x7f86f7384c50>

Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 222, 222, 64)      1792
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 220, 220, 64)      36928
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 110, 110, 64)      0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 108, 108, 128)     73856
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 106, 106, 128)     147584
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 53, 53, 128)       0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 51, 51, 256)       295168
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 49, 49, 256)       590080
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 47, 47, 256)       590080
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 23, 23, 256)       0
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 21, 21, 512)       1180160
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 19, 19, 512)       2359808
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 17, 17, 512)       2359808
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 8, 8, 512)         0
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 6, 6, 512)         2359808
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 4, 4, 512)         2359808
_________________________________________________________________
conv2d_13 (Conv2D)           (None, 2, 2, 512)         2359808
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 1, 1, 512)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 512)               0
_________________________________________________________________
dense_1 (Dense)              (None, 4096)              2101248
_________________________________________________________________
dense_2 (Dense)              (None, 4096)              16781312
_________________________________________________________________
dropout_1 (Dropout)          (None, 4096)              0
_________________________________________________________________
dense_3 (Dense)              (None, 11)                45067
=================================================================
Total params: 33,642,315
Trainable params: 33,642,315
Non-trainable params: 0
_________________________________________________________________
Training and validation:


Epoch 1/25
 - 1410s - loss: 2.3198 - accuracy: 0.1286 - val_loss: 2.2683 - val_accuracy: 0.0000e+00
Epoch 2/25
 - 1411s - loss: 2.2676 - accuracy: 0.1402 - val_loss: 2.2370 - val_accuracy: 0.0000e+00
Epoch 3/25
 - 1410s - loss: 2.2757 - accuracy: 0.1362 - val_loss: 2.5928 - val_accuracy: 0.0000e+00
Epoch 4/25
 - 1414s - loss: 2.2755 - accuracy: 0.1393 - val_loss: 2.3789 - val_accuracy: 0.0000e+00
Epoch 5/25
 - 1417s - loss: 2.2668 - accuracy: 0.1393 - val_loss: 2.2564 - val_accuracy: 0.1000
Epoch 6/25
 - 1410s - loss: 2.2656 - accuracy: 0.1376 - val_loss: 2.2792 - val_accuracy: 0.1000
Epoch 7/25
 - 1414s - loss: 2.2680 - accuracy: 0.1362 - val_loss: 2.2243 - val_accuracy: 0.0000e+00
Epoch 8/25
 - 1429s - loss: 2.2634 - accuracy: 0.1380 - val_loss: 2.3576 - val_accuracy: 0.2000
Epoch 9/25
 - 1430s - loss: 2.2633 - accuracy: 0.1344 - val_loss: 2.2650 - val_accuracy: 0.1000
Epoch 10/25
 - 1432s - loss: 2.2338 - accuracy: 0.1568 - val_loss: 2.1641 - val_accuracy: 0.2000
Epoch 11/25
 - 1423s - loss: 1.8444 - accuracy: 0.3586 - val_loss: 1.9177 - val_accuracy: 0.3000
Epoch 12/25
 - 1421s - loss: 1.2321 - accuracy: 0.5811 - val_loss: 1.6842 - val_accuracy: 0.7000
Epoch 13/25
 - 1427s - loss: 0.9007 - accuracy: 0.7088 - val_loss: 0.6650 - val_accuracy: 0.7000
Epoch 14/25
 - 1426s - loss: 0.6675 - accuracy: 0.7919 - val_loss: 1.2071 - val_accuracy: 0.7000
Epoch 15/25
 - 1428s - loss: 0.5021 - accuracy: 0.8410 - val_loss: 0.9528 - val_accuracy: 0.8000
Epoch 16/25
 - 1420s - loss: 0.4260 - accuracy: 0.8647 - val_loss: 0.4894 - val_accuracy: 0.9000
Epoch 17/25
 - 1417s - loss: 0.3120 - accuracy: 0.9026 - val_loss: 0.0723 - val_accuracy: 1.0000
Epoch 18/25
 - 1452s - loss: 0.2754 - accuracy: 0.9147 - val_loss: 0.0692 - val_accuracy: 1.0000
Epoch 19/25
 - 1456s - loss: 0.2075 - accuracy: 0.9326 - val_loss: 0.8923 - val_accuracy: 0.9000
Epoch 20/25
 - 1459s - loss: 0.1365 - accuracy: 0.9598 - val_loss: 0.1243 - val_accuracy: 0.9000
Epoch 21/25
 - 1450s - loss: 0.1230 - accuracy: 0.9580 - val_loss: 0.8980 - val_accuracy: 0.7000
Epoch 22/25
 - 1446s - loss: 0.1234 - accuracy: 0.9616 - val_loss: 0.9370 - val_accuracy: 0.9000
Epoch 23/25
 - 1447s - loss: 0.1244 - accuracy: 0.9643 - val_loss: 0.4076 - val_accuracy: 0.9000
Epoch 24/25
 - 1661s - loss: 0.0689 - accuracy: 0.9777 - val_loss: 0.0065 - val_accuracy: 1.0000
Epoch 25/25
 - 3115s - loss: 0.0774 - accuracy: 0.9750 - val_loss: 0.9458 - val_accuracy: 0.8000
Accuracy {'val_loss': [2.268286943435669, 2.2369918823242188, 2.5928139686584473, 2.378938674926758, 2.2564361095428467, 2.279184341430664, 2.2243289947509766, 2.357619047164917, 2.2649855613708496, 2.164093017578125, 1.917707085609436, 1.6841846704483032, 0.6649965643882751, 1.207130789756775, 0.9527589082717896, 0.4893873631954193, 0.07229139655828476, 0.06922495365142822, 0.8923214077949524, 0.12433679401874542, 0.8979616165161133, 0.9369570016860962, 0.40759986639022827, 0.0065461741760373116, 0.9457777142524719], 'accuracy': [0.12862885, 0.14024118, 0.13622153, 0.13934793, 0.13934793, 0.13756141, 0.13622153, 0.13800804, 0.13443501, 0.15676641, 0.35864225, 0.581063, 0.7087986, 0.79187137, 0.84100044, 0.8646717, 0.9026351, 0.9146941, 0.9325592, 0.95980346, 0.958017, 0.96159, 0.96426976, 0.9776686, 0.9749888], 'val_accuracy': [0.0, 0.0, 0.0, 0.0, 0.10000000149011612, 0.10000000149011612, 0.0, 0.20000000298023224, 0.10000000149011612, 0.20000000298023224, 0.30000001192092896, 0.699999988079071, 0.699999988079071, 0.699999988079071, 0.800000011920929, 0.8999999761581421, 1.0, 1.0, 0.8999999761581421, 0.8999999761581421, 0.699999988079071, 0.8999999761581421, 0.8999999761581421, 1.0, 0.800000011920929], 'loss': [2.3197945922333623, 2.267504977349353, 2.2756572046573806, 2.2756881760720327, 2.2667992064973084, 2.265623813657262, 2.267884085038881, 2.263443847062493, 2.263236445366884, 2.2338228384369767, 1.844309382696352, 1.2320805285167566, 0.9007019974414447, 0.6674410125050176, 0.5021926196073251, 0.4259740050803902, 0.31212096870242517, 0.2755085043509313, 0.2074166427765115, 0.13644062174417998, 0.12300149501164986, 0.12345285839362612, 0.12445504603472378, 0.06888843739323723, 0.07745091702269567]}


 [0.]


Prediction:



 1/20 [>.............................] - ETA: 34s
 2/20 [==>...........................] - ETA: 19s
 3/20 [===>..........................] - ETA: 13s
 4/20 [=====>........................] - ETA: 10s
 5/20 [======>.......................] - ETA: 8s
 6/20 [========>.....................] - ETA: 7s
 7/20 [=========>....................] - ETA: 6s
 8/20 [===========>..................] - ETA: 6s
 9/20 [============>.................] - ETA: 5s
10/20 [==============>...............] - ETA: 4s
11/20 [===============>..............] - ETA: 4s
12/20 [=================>............] - ETA: 3s
13/20 [==================>...........] - ETA: 3s
14/20 [====================>.........] - ETA: 2s
15/20 [=====================>........] - ETA: 2s
16/20 [=======================>......] - ETA: 1s
17/20 [========================>.....] - ETA: 1s
18/20 [==========================>...] - ETA: 0s
19/20 [===========================>..] - ETA: 0s
20/20 [==============================] - 8s 417ms/step
predictions avec  {'Marine lepen': 9, 'jacques chirac': 5, 'other': 1, 'Brad Pitt': 3, 'jean lassalle': 6, 'Jennifer lopez': 8, 'Donald Trump': 4, 'Bill Gates': 2, 'ali': 0, 'Tom cruise': 10, 'Jean pierre coffe': 7}  :
 [9.0850058e-08 4.2804099e-10 3.2407370e-05 7.7817012e-06 5.2365189e-11
 2.0882776e-15 1.6668184e-09 2.3606063e-03 2.6949569e-05 9.1916188e-12
 8.3632278e-01 1.0764459e-17 8.3608698e-05 4.3249363e-15 2.6734358e-11
 6.5332993e-20 1.0978882e-12 4.8036782e-07 7.1466061e-06 3.9387098e-05]


[9.08500581e-08 1.46301488e-06 1.39587282e-05 1.02745835e-05
 1.16798496e-07 9.99964356e-01 2.99830063e-08 7.13456586e-07
 7.19384952e-09 4.27312186e-09 9.09758637e-06]
[4.2804099e-10 4.0950184e-07 9.9992180e-01 1.0483865e-08 8.2022886e-09
 6.6670218e-05 4.4909698e-09 1.0632063e-05 2.8725466e-07 1.6060565e-07
 1.6556157e-08]
[3.2407370e-05 2.2048354e-03 3.2168154e-02 3.6450345e-02 2.4074435e-03
 9.2477578e-01 9.0148060e-05 1.2562114e-03 1.0328707e-04 2.8385301e-04
 2.2743719e-04]
[7.7817012e-06 2.2607732e-03 8.3338426e-07 1.0037302e-03 2.6198666e-06
 1.0794863e-03 9.2492580e-01 3.6087316e-03 1.2664667e-06 3.7618013e-04
 6.6732839e-02]
[5.2365189e-11 7.0012049e-09 4.0280472e-08 1.3264177e-08 8.8854957e-10
 1.0000000e+00 4.0001343e-11 1.2966153e-09 2.9989277e-12 1.5649341e-11
 8.6849177e-09]
[2.0882776e-15 3.6543465e-08 7.2642879e-13 1.6125092e-10 1.6740325e-16
 4.6939576e-15 3.7650059e-13 5.0992045e-17 1.0000000e+00 6.2819714e-09
 3.5146164e-13]
[1.66681835e-09 2.54833452e-07 1.33056535e-08 4.09928471e-04
 9.16169807e-09 9.99589741e-01 6.08770723e-09 7.44434181e-10
 1.52157540e-11 5.85369184e-11 9.49582315e-08]
[2.3606063e-03 8.2638389e-01 7.1756775e-04 4.4812609e-02 3.3457551e-04
 7.0611556e-04 4.3022931e-03 5.1776544e-05 8.4279232e-02 3.0422984e-02
 5.6283344e-03]
[2.6949569e-05 6.7185545e-03 9.8278564e-01 1.1452513e-03 1.4817290e-03
 2.1493100e-03 4.9833203e-04 3.7734480e-05 2.2278256e-03 2.0482063e-03
 8.8057184e-04]
[9.1916188e-12 4.1320956e-05 1.0032247e-08 1.4091991e-06 6.7517153e-10
 4.6432236e-10 1.0906198e-08 1.8581007e-11 9.8852330e-01 1.1433889e-02
 5.1896336e-09]
[8.36322784e-01 3.60829309e-02 1.52203022e-03 1.89017225e-02
 9.34923519e-05 8.48872587e-02 7.67084444e-03 1.17720934e-04
 1.07120688e-03 4.68125596e-04 1.28618097e-02]
[1.0764459e-17 1.2024675e-08 1.8971439e-13 1.7232410e-11 4.1775172e-17
 7.4023381e-17 5.0641647e-14 1.7005813e-18 9.9999630e-01 3.7145187e-06
 1.5448277e-14]
[8.3608698e-05 3.0663412e-03 2.5645699e-04 5.7126153e-02 1.7224449e-03
 9.0413606e-01 1.2550752e-02 4.3933699e-03 1.7915128e-04 1.2751893e-02
 3.7337907e-03]
[4.3249363e-15 2.5670840e-10 1.2123057e-11 1.0000000e+00 1.1760114e-11
 7.7549949e-09 4.4815038e-10 1.5953249e-11 3.0309159e-13 4.1066112e-14
 3.7803932e-08]
[2.6734358e-11 4.2463174e-07 1.0595539e-08 9.9999833e-01 1.3236759e-08
 7.2991037e-07 5.0629346e-08 2.4236586e-09 4.7439874e-10 9.9986353e-11
 3.8049225e-07]
[6.5332993e-20 4.9190742e-11 2.2236146e-09 9.8202444e-12 9.9999964e-01
 2.9759492e-10 1.9681673e-11 3.3503485e-07 1.5961871e-13 2.7526093e-09
 3.8675465e-12]
[1.0978882e-12 7.3595373e-07 2.0214557e-09 7.8347284e-08 9.0933812e-09
 1.1398907e-09 3.2444746e-08 5.1398508e-10 1.6961711e-05 9.9998224e-01
 9.8266417e-10]
[4.8036782e-07 1.8591261e-04 8.9486473e-04 1.6337501e-02 9.7114051e-04
 5.1404961e-04 5.2416831e-04 3.5632550e-04 4.5789748e-05 3.4062612e-06
 9.8016638e-01]
[7.1466061e-06 4.7269263e-04 7.8950802e-07 2.8358784e-04 1.7609453e-05
 1.8662828e-03 9.9676007e-01 9.4051124e-05 2.2718149e-07 1.1298734e-04
 3.8457024e-04]
[3.9387098e-05 5.6098838e-04 9.8826712e-01 2.4586834e-04 5.1026040e-05
 8.6069014e-04 1.3395643e-04 2.5228185e-03 4.5790146e-03 5.8138568e-04
 2.1577578e-03]
 '''










# No good --> epoch 72 avec training batch -> 20 and test -> 5 (processus killed)
 '''
 Epoch 1/72
2019-11-26 23:03:11.331167: W tensorflow/core/framework/allocator.cc:107] Allocation of 252334080 exceeds 10% of system memory.
2019-11-26 23:03:11.508096: W tensorflow/core/framework/allocator.cc:107] Allocation of 247808000 exceeds 10% of system memory.
2019-11-26 23:03:21.354845: W tensorflow/core/framework/allocator.cc:107] Allocation of 247808000 exceeds 10% of system memory.
2019-11-26 23:03:21.735402: W tensorflow/core/framework/allocator.cc:107] Allocation of 252334080 exceeds 10% of system memory.
2019-11-26 23:03:24.133750: W tensorflow/core/framework/allocator.cc:107] Allocation of 252334080 exceeds 10% of system memory.
 - 1401s - loss: 2.3107 - accuracy: 0.1224 - val_loss: 2.2020 - val_accuracy: 0.0000e+00
Epoch 2/72
 - 1396s - loss: 2.2758 - accuracy: 0.1313 - val_loss: 2.3079 - val_accuracy: 0.0000e+00
Epoch 3/72
 - 1405s - loss: 2.2697 - accuracy: 0.1264 - val_loss: 2.3129 - val_accuracy: 0.2000
Epoch 4/72
 - 1399s - loss: 2.2664 - accuracy: 0.1460 - val_loss: 2.3092 - val_accuracy: 0.0000e+00
Epoch 5/72
 - 1400s - loss: 2.2657 - accuracy: 0.1268 - val_loss: 2.5769 - val_accuracy: 0.0000e+00
Epoch 6/72
 - 1404s - loss: 2.2698 - accuracy: 0.1295 - val_loss: 2.1784 - val_accuracy: 0.2000
Epoch 7/72
 - 1410s - loss: 2.2665 - accuracy: 0.1402 - val_loss: 2.1919 - val_accuracy: 0.4000
Epoch 8/72
 - 1419s - loss: 2.2679 - accuracy: 0.1300 - val_loss: 2.2964 - val_accuracy: 0.2000
Epoch 9/72
 - 1426s - loss: 2.2674 - accuracy: 0.1367 - val_loss: 2.2560 - val_accuracy: 0.0000e+00
Epoch 10/72
 - 1420s - loss: 2.2630 - accuracy: 0.1425 - val_loss: 2.1499 - val_accuracy: 0.0000e+00
Epoch 11/72
 - 1409s - loss: 2.2677 - accuracy: 0.1304 - val_loss: 2.1463 - val_accuracy: 0.4000
Epoch 12/72
 - 1415s - loss: 2.2666 - accuracy: 0.1331 - val_loss: 2.6241 - val_accuracy: 0.0000e+00
Epoch 13/72
 - 1417s - loss: 2.2670 - accuracy: 0.1380 - val_loss: 2.5088 - val_accuracy: 0.2000
Epoch 14/72
 - 1427s - loss: 2.2674 - accuracy: 0.1318 - val_loss: 2.2443 - val_accuracy: 0.2000
Epoch 15/72
 - 1427s - loss: 2.2654 - accuracy: 0.1291 - val_loss: 2.2849 - val_accuracy: 0.2000
Epoch 16/72
 - 1422s - loss: 2.2644 - accuracy: 0.1398 - val_loss: 2.3292 - val_accuracy: 0.0000e+00
Epoch 17/72
 - 1420s - loss: 2.2645 - accuracy: 0.1367 - val_loss: 2.3211 - val_accuracy: 0.0000e+00
Epoch 18/72
 - 1423s - loss: 2.2662 - accuracy: 0.1429 - val_loss: 2.3335 - val_accuracy: 0.0000e+00
Epoch 19/72
 - 1432s - loss: 2.2662 - accuracy: 0.1304 - val_loss: 2.3171 - val_accuracy: 0.0000e+00
Epoch 20/72
 - 1434s - loss: 2.2656 - accuracy: 0.1385 - val_loss: 2.2554 - val_accuracy: 0.2000
Epoch 21/72
 - 1437s - loss: 2.2659 - accuracy: 0.1313 - val_loss: 2.8729 - val_accuracy: 0.0000e+00
Epoch 22/72
 - 1417s - loss: 2.2629 - accuracy: 0.1411 - val_loss: 2.2364 - val_accuracy: 0.2000
Epoch 23/72
 - 1419s - loss: 2.2639 - accuracy: 0.1456 - val_loss: 2.3282 - val_accuracy: 0.0000e+00
Epoch 24/72
 - 1427s - loss: 2.2609 - accuracy: 0.1362 - val_loss: 2.3451 - val_accuracy: 0.8000
Epoch 25/72
 - 1431s - loss: 2.2636 - accuracy: 0.1393 - val_loss: 2.3016 - val_accuracy: 0.2000
Epoch 26/72
 - 1429s - loss: 2.2642 - accuracy: 0.1393 - val_loss: 2.8160 - val_accuracy: 0.0000e+00
Epoch 27/72
 - 1422s - loss: 2.2648 - accuracy: 0.1331 - val_loss: 2.2896 - val_accuracy: 0.0000e+00
Epoch 28/72
 - 1423s - loss: 2.2640 - accuracy: 0.1353 - val_loss: 2.2869 - val_accuracy: 0.0000e+00
Epoch 29/72
 - 1428s - loss: 2.2650 - accuracy: 0.1411 - val_loss: 2.3359 - val_accuracy: 0.0000e+00
Epoch 30/72
 - 1423s - loss: 2.2657 - accuracy: 0.1420 - val_loss: 2.1997 - val_accuracy: 0.0000e+00
Epoch 31/72
 - 1435s - loss: 2.2651 - accuracy: 0.1353 - val_loss: 2.2415 - val_accuracy: 0.0000e+00
Epoch 32/72
'''

# Le meilleur --> epoch 38, batch training 15 et valid -> 10
# recognize_4class_1_0.5dropoutPlus.h5
'''
Found 2239 images belonging to 11 classes.
Found 356 images belonging to 11 classes.
Found 63 images belonging to 11 classes.
predictions avec  {'jacques chirac': 5, 'Tom cruise': 10, 'ali': 0, 'jean lassalle': 6, 'Jennifer lopez': 8, 'Bill Gates': 2, 'Brad Pitt': 3, 'Jean pierre coffe': 7, 'Donald Trump': 4, 'other': 1, 'Marine lepen': 9}  :  <keras.preprocessing.image.DirectoryIterator object at 0x7f86608fbbe0>

WARNING:tensorflow:From /home/gespenst/.local/lib/python3.5/site-packages/keras/backend/tensorflow_backend.py:4070: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.

Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 222, 222, 64)      1792
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 220, 220, 64)      36928
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 110, 110, 64)      0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 108, 108, 128)     73856
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 106, 106, 128)     147584
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 53, 53, 128)       0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 51, 51, 256)       295168
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 49, 49, 256)       590080
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 47, 47, 256)       590080
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 23, 23, 256)       0
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 21, 21, 512)       1180160
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 19, 19, 512)       2359808
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 17, 17, 512)       2359808
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 8, 8, 512)         0
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 6, 6, 512)         2359808
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 4, 4, 512)         2359808
_________________________________________________________________
conv2d_13 (Conv2D)           (None, 2, 2, 512)         2359808
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 1, 1, 512)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 512)               0
_________________________________________________________________
dense_1 (Dense)              (None, 4096)              2101248
_________________________________________________________________
dense_2 (Dense)              (None, 4096)              16781312
_________________________________________________________________
dropout_1 (Dropout)          (None, 4096)              0
_________________________________________________________________
dense_3 (Dense)              (None, 11)                45067
=================================================================
Total params: 33,642,315
Trainable params: 33,642,315
Non-trainable params: 0
_________________________________________________________________
Training and validation:


2019-11-27 11:29:15.949242: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-11-27 11:29:16.253580: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2793620000 Hz
2019-11-27 11:29:16.259872: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x555cb7dd9900 executing computations on platform Host. Devices:
2019-11-27 11:29:16.259907: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): <undefined>, <undefined>
2019-11-27 11:29:16.777651: W tensorflow/compiler/jit/mark_for_compilation_pass.cc:1412] (One-time warning): Not using XLA:CPU for cluster because envvar TF_XLA_FLAGS=--tf_xla_cpu_global_jit was not set.  If you want XLA:CPU, either set that envvar, or use experimental_jit_scope to enable XLA:CPU.  To confirm that XLA is active, pass --vmodule=xla_compilation_cache=1 (as a proper command-line flag, not via TF_XLA_FLAGS) or set the envvar XLA_FLAGS=--xla_hlo_profile.
WARNING:tensorflow:From /home/gespenst/.local/lib/python3.5/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/38
 - 1448s - loss: 2.3011 - accuracy: 0.1268 - val_loss: 2.1878 - val_accuracy: 0.2000
Epoch 2/38
 - 1434s - loss: 2.2742 - accuracy: 0.1268 - val_loss: 2.1415 - val_accuracy: 0.2000
Epoch 3/38
 - 1456s - loss: 2.2777 - accuracy: 0.1393 - val_loss: 2.2881 - val_accuracy: 0.0000e+00
Epoch 4/38
 - 1454s - loss: 2.2660 - accuracy: 0.1291 - val_loss: 2.2426 - val_accuracy: 0.1000
Epoch 5/38
 - 1448s - loss: 2.2782 - accuracy: 0.1309 - val_loss: 2.2635 - val_accuracy: 0.0000e+00
Epoch 6/38
 - 1446s - loss: 2.2222 - accuracy: 0.1514 - val_loss: 2.2862 - val_accuracy: 0.1000
Epoch 7/38
 - 1445s - loss: 1.8346 - accuracy: 0.3149 - val_loss: 1.6504 - val_accuracy: 0.3000
Epoch 8/38
 - 1442s - loss: 1.4118 - accuracy: 0.5074 - val_loss: 1.6008 - val_accuracy: 0.4000
Epoch 9/38
 - 1439s - loss: 1.0083 - accuracy: 0.6883 - val_loss: 1.0366 - val_accuracy: 0.8000
Epoch 10/38
 - 1443s - loss: 0.6600 - accuracy: 0.7959 - val_loss: 0.4848 - val_accuracy: 0.8000
Epoch 11/38
 - 1445s - loss: 0.4435 - accuracy: 0.8557 - val_loss: 0.0360 - val_accuracy: 1.0000
Epoch 12/38
 - 1448s - loss: 0.3506 - accuracy: 0.8888 - val_loss: 0.0119 - val_accuracy: 1.0000
Epoch 13/38
 - 1442s - loss: 0.2902 - accuracy: 0.9169 - val_loss: 0.0519 - val_accuracy: 1.0000
Epoch 14/38
 - 1448s - loss: 0.2225 - accuracy: 0.9268 - val_loss: 0.2126 - val_accuracy: 1.0000
Epoch 15/38
 - 1451s - loss: 0.1556 - accuracy: 0.9482 - val_loss: 0.2044 - val_accuracy: 0.9000
Epoch 16/38
 - 1450s - loss: 0.1159 - accuracy: 0.9603 - val_loss: 0.0071 - val_accuracy: 1.0000
Epoch 17/38
 - 1449s - loss: 0.1239 - accuracy: 0.9665 - val_loss: 0.0619 - val_accuracy: 1.0000
Epoch 18/38
 - 1448s - loss: 0.0762 - accuracy: 0.9723 - val_loss: 0.5487 - val_accuracy: 0.9000
Epoch 19/38
 - 1445s - loss: 0.0721 - accuracy: 0.9803 - val_loss: 0.5503 - val_accuracy: 0.8000
Epoch 20/38
 - 1444s - loss: 0.0421 - accuracy: 0.9884 - val_loss: 0.0034 - val_accuracy: 1.0000
Epoch 21/38
 - 1446s - loss: 0.1106 - accuracy: 0.9669 - val_loss: 0.9330 - val_accuracy: 0.8000
Epoch 22/38
 - 1441s - loss: 0.0269 - accuracy: 0.9924 - val_loss: 0.8724 - val_accuracy: 0.9000
Epoch 23/38
 - 1442s - loss: 0.0886 - accuracy: 0.9732 - val_loss: 0.2326 - val_accuracy: 0.9000
Epoch 24/38
 - 1439s - loss: 0.0797 - accuracy: 0.9736 - val_loss: 0.1338 - val_accuracy: 0.9000
Epoch 25/38
 - 1438s - loss: 0.0579 - accuracy: 0.9866 - val_loss: 0.2781 - val_accuracy: 0.9000
Epoch 26/38
 - 1438s - loss: 0.0138 - accuracy: 0.9955 - val_loss: 2.4676e-06 - val_accuracy: 1.0000
Epoch 27/38
 - 1440s - loss: 0.0311 - accuracy: 0.9937 - val_loss: 6.1392e-06 - val_accuracy: 1.0000
Epoch 28/38
 - 1442s - loss: 0.0253 - accuracy: 0.9933 - val_loss: 2.1244 - val_accuracy: 0.7000
Epoch 29/38
 - 1439s - loss: 0.0125 - accuracy: 0.9969 - val_loss: 0.1204 - val_accuracy: 0.9000
Epoch 30/38
 - 1451s - loss: 0.0179 - accuracy: 0.9955 - val_loss: 1.1791 - val_accuracy: 0.8000
Epoch 31/38
 - 1440s - loss: 0.0022 - accuracy: 0.9996 - val_loss: 4.1727 - val_accuracy: 0.7000
Epoch 32/38
 - 1432s - loss: 0.0771 - accuracy: 0.9795 - val_loss: 1.3693 - val_accuracy: 0.7000
Epoch 33/38
 - 1422s - loss: 0.0285 - accuracy: 0.9920 - val_loss: 2.3183 - val_accuracy: 0.7000
Epoch 34/38
 - 1424s - loss: 0.0541 - accuracy: 0.9915 - val_loss: 1.3065 - val_accuracy: 0.8000
Epoch 35/38
 - 1425s - loss: 0.0404 - accuracy: 0.9884 - val_loss: 2.4673 - val_accuracy: 0.6000
Epoch 36/38
 - 1424s - loss: 0.0287 - accuracy: 0.9920 - val_loss: 0.8505 - val_accuracy: 0.8333
Epoch 37/38
 - 1424s - loss: 7.9239e-04 - accuracy: 1.0000 - val_loss: 0.6423 - val_accuracy: 0.9000
Epoch 38/38
 - 1424s - loss: 1.7491e-04 - accuracy: 1.0000 - val_loss: 1.1802e-06 - val_accuracy: 1.0000
Accuracy {'val_loss': [2.1878135204315186, 2.141477108001709, 2.288083553314209, 2.242572069168091, 2.2634782791137695, 2.286219358444214, 1.6503667831420898, 1.6008161306381226, 1.0365618467330933, 0.48483437299728394, 0.03602081537246704, 0.011911911889910698, 0.05188099294900894, 0.2125857174396515, 0.20443598926067352, 0.007125875912606716, 0.0619029775261879, 0.5487250685691833, 0.5503034591674805, 0.0034269809257239103, 0.9330438375473022, 0.8724043965339661, 0.23264005780220032, 0.13378240168094635, 0.27813127636909485, 2.4676110115251504e-06, 6.13919928582618e-06, 2.1244256496429443, 0.12041487544775009, 1.1791326999664307, 4.172654151916504, 1.3693103790283203, 2.318343162536621, 1.306525468826294, 2.4673256874084473, 0.8505372405052185, 0.6422601938247681, 1.1801685104728676e-06], 'val_accuracy': [0.20000000298023224, 0.20000000298023224, 0.0, 0.10000000149011612, 0.0, 0.10000000149011612, 0.30000001192092896, 0.4000000059604645, 0.800000011920929, 0.800000011920929, 1.0, 1.0, 1.0, 1.0, 0.8999999761581421, 1.0, 1.0, 0.8999999761581421, 0.800000011920929, 1.0, 0.800000011920929, 0.8999999761581421, 0.8999999761581421, 0.8999999761581421, 0.8999999761581421, 1.0, 1.0, 0.699999988079071, 0.8999999761581421, 0.800000011920929, 0.699999988079071, 0.699999988079071, 0.699999988079071, 0.800000011920929, 0.6000000238418579, 0.8333333134651184, 0.8999999761581421, 1.0], 'loss': [2.301252175463582, 2.2745458614405587, 2.2777206715647265, 2.266426915087408, 2.27780632346589, 2.2215758446020013, 1.8347971361627957, 1.4130609473675264, 1.0095916286929476, 0.6596749173514728, 0.44562847083480217, 0.3490784886135466, 0.2851072320928394, 0.22355769921402818, 0.1562805707318583, 0.1164556418872439, 0.12263069502831167, 0.07654084611080385, 0.07241607963462855, 0.04232261043915079, 0.1111436671779756, 0.0267848188747484, 0.08889105602947556, 0.07649800449805398, 0.058225081622415506, 0.013878612415364323, 0.031210899967077488, 0.025445992717469033, 0.012594099520802903, 0.017992195310105704, 0.002221011796075793, 0.07752009981804242, 0.028640748665601342, 0.053869517037683265, 0.04059429646288059, 0.028857450679459582, 0.0007962262157489588, 0.00017412640627783308], 'accuracy': [0.12684233, 0.12684233, 0.13934793, 0.12907548, 0.130862, 0.15140688, 0.3148727, 0.50736934, 0.6882537, 0.79589105, 0.8557392, 0.88878965, 0.9169272, 0.92675304, 0.94819117, 0.96025014, 0.9665029, 0.97230905, 0.98034835, 0.98838764, 0.9669495, 0.9924073, 0.97320235, 0.97364897, 0.9866012, 0.9955337, 0.99374723, 0.99330056, 0.9968736, 0.9955337, 0.9995534, 0.9794551, 0.9919607, 0.9915141, 0.98838764, 0.9919607, 1.0, 1.0]}

'''
