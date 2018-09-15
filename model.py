import csv
import cv2
import numpy as np
import sklearn

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Conv2D, Convolution2D, MaxPooling2D, Cropping2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

BATCH_SIZE = 32

def import_driving_log():
	lines = []
	images = []
	steering_angles = []
	
	# Import images from left/center/right cameras and their respective steering angle
	with open('MyData/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		iterate_reader = iter(reader)
		next(iterate_reader)
		for index, line in enumerate(iterate_reader):
			img_center = cv2.imread('MyData/IMG/' + line[0].split('/')[-1])
			img_left = cv2.imread('MyData/IMG/' + line[1].split('/')[-1])
			img_right = cv2.imread('MyData/IMG/' + line[2].split('/')[-1])
			#img_center_resized = cv2.resize(img_center, dsize=(int(img_center.shape[1]/4), int(img_center.shape[0]/4)), interpolation=cv2.INTER_CUBIC)
			#img_left_resized = cv2.resize(img_left, dsize=(int(img_left.shape[1]/4), int(img_left.shape[0]/4)), interpolation=cv2.INTER_CUBIC)
			#img_right_resized = cv2.resize(img_right, dsize=(int(img_right.shape[1]/4), int(img_right.shape[0]/4)), interpolation=cv2.INTER_CUBIC)
			
			images.append(img_center)
			images.append(img_left)	
			images.append(img_right)
			
			steering_center = float(line[3])
			steering_left = steering_center + 0.2
			steering_right = steering_center - 0.2
			
			steering_angles.append(steering_center)
			steering_angles.append(steering_left)
			steering_angles.append(steering_right)

	X_train = np.array(images)
	y_train = np.array(steering_angles)
	
	return X_train, y_train

def flip_images(X_train, y_train):
	# Flip images to get more data for the opposite steering
	augmented_images, augmented_measurements = [], []
	for image, steering_angles in zip(X_train, y_train):
		augmented_images.append(image)
		augmented_measurements.append(steering_angles)
		augmented_images.append(cv2.flip(image,1))
		augmented_measurements.append(steering_angles*-1.0)

	X_train = np.array(augmented_images)
	y_train = np.array(augmented_measurements)
	
	return X_train, y_train

def create_cnn_model():
	# Create Sequential model (linear stack of layers) for CNN
	model = Sequential()

	# Normalize input images
	model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3), output_shape=(160,320,3)))

	# Crop images (70 pixels on top, 25 pixels on bottom) to get rid of part of the images bringing no relevant information (sky, hood of the car)
	model.add(Cropping2D(cropping=((70,25), (0,0))))
	
	# Add layers to the neural network (based on the architecture of NVIDIA's CNN)
	model.add(Conv2D(18,5,5, activation="relu"))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(24,3,3, activation="relu"))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(48,3,3, activation="relu"))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(96,3,3, activation="relu"))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(120))
	model.add(Dropout(0.4))
	model.add(Dense(84))
	model.add(Dropout(0.4))
	model.add(Dense(1))
	
	model.summary()
	
	return model

# Use a generator to process part of the dataset (shuffled) on the fly only when needed, which is more memory-efficient.
def generator(X, y, batch_size = BATCH_SIZE):
	assert len(X) == len(y)
	num_samples = len(X)
	while 1:
		X_shuffled, y_shuffled = shuffle(X, y, random_state=0)
		for offset in range(0, num_samples, batch_size):
			images_samples = X_shuffled[offset:offset+batch_size]
			angles_samples = y_shuffled[offset:offset+batch_size]
			X_generated = np.array(images_samples)
			y_generated = np.array(angles_samples)
			yield shuffle(X_generated, y_generated)
	

# import the information from the dataset (images as input and steering angles as output)
X_train, y_train = import_driving_log()
# create more data by adding flipped images of the same dataset
X_train, y_train = flip_images(X_train, y_train)

print(X_train.shape)

# Split the dataset between training samples (80%) and validation samples (20%) 
X_train_samples, X_validation_samples, y_train_samples, y_validation_samples = train_test_split(X_train, y_train, test_size=0.2)

print(X_train_samples.shape)
print(X_validation_samples.shape)

# compile and train the model using the generator function
train_generator = generator(X_train_samples, y_train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(X_validation_samples, y_validation_samples, batch_size=BATCH_SIZE)

# create neural network based on our model architecture
model = create_cnn_model()

# Configure the learning process (using adam optimization algorithm)
model.compile(loss='mse', optimizer='adam')

# Trains the model on data generated batch-by-batch by the generator
model.fit_generator(train_generator, steps_per_epoch= len(X_train_samples)/BATCH_SIZE, validation_data=validation_generator, validation_steps=len(X_validation_samples)/BATCH_SIZE, epochs=5, verbose = 1)

model.save('model.h5')