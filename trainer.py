import os
TRAINING_DIR = ""
dir = TRAINING_DIR
A_dir = os.path.join (dir+"A")
B_dir = os.path.join (dir+"B")
C_dir = os.path.join (dir+"C")
D_dir = os.path.join (dir+"D")
E_dir = os.path.join (dir+"E")
F_dir = os.path.join (dir+"F")
G_dir = os.path.join (dir+"G")
H_dir = os.path.join (dir+"H")
J_dir = os.path.join (dir+"J")
K_dir = os.path.join (dir+"K")
L_dir = os.path.join (dir+"L")
M_dir = os.path.join (dir+"M")
N_dir = os.path.join (dir+"N")
O_dir = os.path.join (dir+"O")
P_dir = os.path.join (dir+"P")
R_dir = os.path.join (dir+"R")
S_dir = os.path.join (dir+"S")
T_dir = os.path.join (dir+"T")
U_dir = os.path.join (dir+"U")
V_dir = os.path.join (dir+"V")
W_dir = os.path.join (dir+"W")
X_dir = os.path.join (dir+"X")
Z_dir = os.path.join (dir+"Z")

A_files = os.listdir(A_dir)
B_files = os.listdir(B_dir)
C_files = os.listdir(C_dir)
D_files = os.listdir(D_dir)
E_files = os.listdir(E_dir)
F_files = os.listdir(F_dir)
G_files = os.listdir(G_dir)
H_files = os.listdir(H_dir)
J_files = os.listdir(J_dir)
K_files = os.listdir(K_dir)
L_files = os.listdir(L_dir)
M_files = os.listdir(M_dir)
N_files = os.listdir(N_dir)
O_files = os.listdir(O_dir)
P_files = os.listdir(P_dir)
R_files = os.listdir(R_dir)
S_files = os.listdir(S_dir)
T_files = os.listdir(T_dir)
U_files = os.listdir(U_dir)
V_files = os.listdir(V_dir)
W_files = os.listdir(W_dir)
X_files = os.listdir(X_dir)
Z_files = os.listdir(Z_dir)
print(len(A_files))
print(len(B_files))
print(len(C_files))
print(len(D_files))
print(len(E_files))
print(len(F_files))
print(len(G_files))
print(len(H_files))
print(len(J_files))
print(len(K_files))
print(len(L_files))
print(len(M_files))
print(len(N_files))
print(len(O_files))
print(len(P_files))
print(len(R_files))
print(len(S_files))
print(len(T_files))
print(len(U_files))
print(len(V_files))
print(len(W_files))
print(len(X_files))
print(len(Z_files))

import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

training_datagen = ImageDataGenerator(
    rescale=1. / 255,
    fill_mode='nearest'
)
train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(40, 30),
    class_mode='categorical'
)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(40, 30, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(23, activation='softmax')
])

model.summary()
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    train_generator,
    epochs=75
)
model.save("fmodel.h5")

