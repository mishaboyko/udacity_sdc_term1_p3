import csv
import cv2
import numpy as np

# set variables
lines = []
recordings_path = '/home/self_driving_car_ND/term1/p3/CarND-Behavioral-Cloning-P3-master/recordings/'
csv_name = 'driving_log.csv'
path_images = 'some_custom_path'

path_prefect_line = recordings_path + 'perfect_line/' + csv_name
path_recovery_right = recordings_path + 'recovery_right/' + csv_name
path_recovery_left = recordings_path + 'recovery_left/' + csv_name


'''
Read lines from a file.
'''
def load_image_paths(path_csv):
    print("Processing" + path_csv)
    lines_buff = []
    with open(path_csv, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines_buff.append(line)
    return lines_buff

# collect all images for training and validation
lines.extend(load_image_paths(path_prefect_line))
print("Lines[] contains {} images".format(len(lines)))
lines.extend(load_image_paths(path_recovery_right))
print("Lines[] contains {} images".format(len(lines)))
lines.extend(load_image_paths(path_recovery_left))
print("Lines[] contains {} images".format(len(lines)))

images = []
measurements = []

''''
Make use of all 3 camera outputs:
- For left and right camera images, manipulate the measurement correspondingly +/- 0.2 steering degrees.  
'''
for line in lines:
    # uncomment this if you want to change default path to the images
    #image_center = cv2.imread(path_images + line[0].split('/')[-1])
    #image_left = cv2.imread(path_images + line[1].split('/')[-1])
    #image_right = cv2.imread(path_images + line[2].split('/')[-1])
    image_center = cv2.imread(line[0])
    image_left = cv2.imread(line[1])
    image_right = cv2.imread(line[2])

    steering_angle_correction = 0.2
    measurement = float(line[3])
    steering_angle_left = measurement + steering_angle_correction
    steering_angle_right = measurement - steering_angle_correction

    images.append(image_center)
    images.append(image_left)
    images.append(image_right)

    measurements.append(measurement)
    measurements.append(steering_angle_left)
    measurements.append(steering_angle_right)

''''
Generate additional training data by horizontally flipping original images and changing the sign of the measurement
'''
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    # flip images horizontally
    augmented_images.append(cv2.flip(image, 1))
    # invert the steering angle
    augmented_measurements.append(-measurement)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D

# Fully Connected Neural Network in Keras, taken from Nvidia
model = Sequential()

# Preprocessing 1: Normalize the image to the range [0:1]
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# Preprocessing 2: Chop top and bottom of the image
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
# Layers 1-3: Convolutional layers with 24/36/48 filters, a 5x5 kernel, 2x2 strides and ReLU activation function
model.add(Convolution2D(24, (5, 5), activation="relu", strides=(2,2)))
model.add(Convolution2D(36, (5, 5), activation="relu", strides=(2,2)))
model.add(Convolution2D(48, (5, 5), activation="relu", strides=(2,2)))
# Layers 4-5: Convolutional layers with 64 filters, a 3x3 kernel and valid padding
model.add(Convolution2D(64, (3, 3), activation="relu"))
model.add(Convolution2D(64, (3, 3), activation="relu"))
# Layers 6: Flatting
model.add(Flatten())
# Layer 7-10: Dense() with an output width of 100/50/10/1.
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=5, shuffle=True, validation_split=0.2)

model.save('model.h5')
exit()
