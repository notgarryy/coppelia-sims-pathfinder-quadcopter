import pickle
import time
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# Load training data shape (optional if needed for consistency)
data = np.load('C:/Users/ASUS/Documents/Code/CoppeliaSims_py/quadcopter_path_finding/data/data.npy')
target = np.load('C:/Users/ASUS/Documents/Code/CoppeliaSims_py/quadcopter_path_finding/data/target.npy')

# === Define model architecture (must match training) ===
model = Sequential()
model.add(Conv2D(256, (3, 3), input_shape=(50, 50, 1), name='conv2d'))
model.add(Activation('relu', name='activation'))
model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling2d'))

model.add(Conv2D(128, (3, 3), name='conv2d_1'))
model.add(Activation('relu', name='activation_1'))
model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_1'))

model.add(Flatten(name='flatten'))
model.add(Dropout(0.5, name='dropout'))
model.add(Dense(2048, activation='relu', name='dense'))
model.add(Dropout(0.5, name='dropout_1'))
model.add(Dense(256, activation='relu', name='dense_1'))
model.add(Dropout(0.5, name='dropout_2'))
model.add(Dense(128, activation='relu', name='dense_2'))
model.add(Dropout(0.5, name='dropout_3'))
model.add(Dense(64, activation='relu', name='dense_3'))
model.add(Dense(6, activation='softmax', name='dense_4'))

# Compile and build the model before loading weights
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model(np.zeros((1, 50, 50, 1)))  # Build the model

# === Load pretrained weights ===
model.load_weights("C:/Users/ASUS/Documents/Code/CoppeliaSims_py/quadcopter_path_finding/model/model.weights.h5")

# === Load class label dictionary ===
with open("C:/Users/ASUS/Documents/Code/CoppeliaSims_py/quadcopter_path_finding/data/data_training.pkl", "rb") as dict_file:
    category_dict = pickle.load(dict_file)

# === Connect to CoppeliaSim ===
client = RemoteAPIClient()
sim = client.require('sim')

sim.startSimulation()

quadcopter_target = sim.getObject('/target')
camera = sim.getObject('/Quadcopter/visionSensor')

position = sim.getObjectPosition(quadcopter_target, -1)
x, y, z = position
delta = 0.1

try:
    while True:
        img, [resX, resY] = sim.getVisionSensorImg(camera)
        img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
        img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0)
        img = cv2.resize(img, (512, 512))

        test_img = cv2.resize(img, (50, 50))
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        test_img = test_img / 255.0
        test_img = test_img.reshape(1, 50, 50, 1)

        results = model.predict(test_img)
        label = np.argmax(results, axis=1)[0]
        acc = int(np.max(results, axis=1)[0] * 100)

        print(f"Moving: {category_dict[label]} with {acc}% accuracy.")

        # Update position
        if label == 0:
            z -= delta
        elif label == 1:
            x += delta
        elif label == 2:
            y += delta
        elif label == 3:
            y -= delta
        elif label == 4:
            pass  # Stay
        elif label == 5:
            z += delta

        sim.setObjectPosition(quadcopter_target, -1, [x, y, z])

        cv2.imshow("Quadcopter View", img)
        if cv2.waitKey(1) == ord('q'):
            break

        time.sleep(0.01)

except Exception as e:
    print(f"Error: {e}")

# Cleanup
cv2.destroyAllWindows()
sim.stopSimulation()
print("Simulation stopped")
