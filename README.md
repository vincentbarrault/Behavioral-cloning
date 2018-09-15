# Behavioral cloning (using Convolutional Neural Network)

This goal of this project is to train a convolutional neural network to make a car drive on a track as part of a driving simulator. All the project was made in Python, and the Neural network was created by using Keras API (TensorFlow implementation).

The steps of this project are:
- Collecting training dataset,
- Preprocessing images from the dataset,
- Design, train and evaluate a model architecture based on convolutional neural network,
- Use the model to test the solution and make the car drive autonomously on the simulator track.

| File | Description |
|--|--|
| model.py |Script to implement model architecture and train CNN |
| model.h5 | Output of model.py storing model weights. |
| drive.py  | Script to drive the car |
| video.py | Script to create the video recording when driving simulator is in autonomous mode. |
| video.mp4 | Video recording of your vehicle driving autonomously more than 1 lap of the track |

![Video recording of vehicule driving](https://github.com/vincentbarrault/Behavioral-cloning/blob/master/Resource/video.gif?raw=true)

## Dataset collection

### Collecting Datas

To create the dataset, the "training mode" of the driving simulator was used. The car was driven during 2 full laps, trying to stay as much as possible in the middle of the road. Additional datas were collected separately for harder parts of the track.

### Summary of Data Set

The datasets consist of 5937 images (1979 for each camera: center, left, right) and their respective steering angles. The size of the dataset was doubled by adding the "flipped" version of each image (with the opposite steering value), giving a dataset size of 11874, of which:

-   Training set (80%) : 9499 images
-   Validation set (20%) : 2375 images


The images and the driving log (containing the path for images as well as values for steering angle, speed...) are not on this repo because of their size.

## Preprocess Data Set

Before providing the images as input to the neural network, the data set needs to be preprocessed in order to make the recognition by the algorithm easier and/or faster.

### Normalizing Data Set

The pixel value of an image is between 0 and 255. Normalizing an image can be done in several ways, but one of the easiest way is to convert the pixel values so that their range goes from -0.5 to 0.5.

    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3), output_shape=(160,320,3)))

This normalization can help in case of images with a poor constrast, or for removing noise but at the same time bring the image into a range of intensity values that is "normal" (meaning that, statistically, it follows a normal distribution as much as possible)

### Cropping images

The cameras in the simulator capture 160 pixel by 320 pixel images. The top and bottom portions of the image capture not so relevant information such as trees, sky or hood of the car.

The model might train faster by cropping each image to focus on only the portion of the image that is useful for predicting a steering angle:

    model.add(Cropping2D(cropping=((70,25), (0,0))))

## Model Architecture

The architecture chosen is based on NVIDIA's Convolution Neural Network described in the following article:
https://devblogs.nvidia.com/deep-learning-self-driving-cars/

The architecture is shown in the picture below:

![NVIDIA CNN](https://devblogs.nvidia.com/wp-content/uploads/2016/08/cnn-architecture.png)

For this project, the main differences with the original NVIDIA architecture are:
- After each 2D Convolution layer, the CNN performs maximum pooling to reduce the dimensions of the image.
- After performing full connection (Layers 12 and 14), a dropout of 0.4 is performed on the outputs of the previous neural network layer, to also overcome the problem of overﬁtting,

The following table shows the layers used in the model architecture of the project:

|  | Layer type | Output | Number of parameters |
|--|--|--|--|
| 1 | Lambda (normalizing) | (160, 320, 3) | 0 |
| 2 | Cropping | (65, 320, 3) | 0 |
| 3 | 2D Convolution  | (61, 316, 18) | 1368 |
| 4 | Max Pooling | (30, 158, 18)  | 0 |
| 5 | 2D Convolution  | (28, 156, 24)  | 3912 |
| 6 | Max Pooling | (14, 78, 24)    | 0 |
| 7 | 2D Convolution  | (12, 76, 48) | 10416 |
| 8 | Max Pooling | (6, 38, 48)   | 0 |
| 9 | 2D Convolution  | (4, 36, 96) | 41568 |
| 10 | Max Pooling | (2, 18, 96)  | 0 |
| 11 | Flatten  | 3456 | 0 |
| 12 | Dense | 120  | 414840 |
| 13 | Dropout  | 120 | 1368 |
| 14 | Dense | 84  | 10164 |
| 15 | Dropout  | 84 | 1368 |
| 16 | Dense | 1  | 85 |



### Discussion on model architecture

### Layers
This network is a convolutional neural network, as these tend to do very well with images. The architecture is similar to NVIDIA "End-to-End Deep Learning for Self-Driving Cars" neural network, with 5 convolutional layers and 3 fully connected layers.

#### Parameters
The model was trained, then evaluated by using:
- Adam optimizer: computationally efficient and well suited for large scale problems (data or parameters), this optimizer has a broad adoption for deep learning applications in computer vision, extending the "classic" stochastic gradient descent algorithm,
- Batch size of **32**,
- Epoch of **5**:  1 Epoch corresponds to the complete (validation here) data set being passed forward and backward through the neural network. 

Running training/evaluation with a higher number of epoch does not affect considerably the validation accuracy (see below, the loss on the training set and on the validation set after each epoch). The loss on the validation set after **5** epochs is **0,285** and after **15** epochs **0,247**. The same test was also made with a higher number of samples per epoch, this had almost no impact or improvement when letting the car drive autonomously around the track. The decision was taken to use **5** epochs.


#### Training and validation of our model

##### With 5 Epochs:

> Epoch 1/5
> 297/296 [======================] - 22s - loss: 0.0377 - val_loss: 0.0341
> 
> Epoch 2/5
> 297/296 [======================] - 15s - loss: 0.0316 - val_loss: 0.0316
> 
> Epoch 3/5
> 297/296 [======================] - 16s - loss: 0.0297 - val_loss: 0.0300
> 
> Epoch 4/5
> 297/296 [======================] - 16s - loss: 0.0281 - val_loss: 0.0301
> 
> Epoch 5/5
> 297/296 [======================] - 15s - loss: 0.0273 - val_loss: 0.0285

##### With 15 Epochs:

> Epoch 1/15
> 297/296 [======================] - 27s - loss: 0.0389 - val_loss: 0.0265
> 
> Epoch 2/15
> 297/296 [======================] - 22s - loss: 0.0327 - val_loss: 0.0286
> 
> Epoch 3/15
> 297/296 [======================] - 22s - loss: 0.0306 - val_loss: 0.0286
> 
> Epoch 4/15
> 297/296 [======================] - 22s - loss: 0.0292 - val_loss: 0.0255
> 
> Epoch 5/15
> 297/296 [======================] - 22s - loss: 0.0280 - val_loss: 0.0270
> 
> Epoch 6/15
> 297/296 [======================] - 22s - loss: 0.0273 - val_loss: 0.0257
> 
> Epoch 7/15
> 297/296 [======================] - 22s - loss: 0.0264 - val_loss: 0.0268
> 
> Epoch 8/15
> 297/296 [======================] - 22s - loss: 0.0260 - val_loss: 0.0242
> 
> Epoch 9/15
> 297/296 [======================] - 22s - loss: 0.0245 - val_loss: 0.0241
> 
> Epoch 10/15
> 297/296 [======================]- 22s - loss: 0.0243 - val_loss: 0.0241
> 
> Epoch 11/15
> 297/296 [======================] - 22s - loss: 0.0233 - val_loss: 0.0247
> 
> Epoch 12/15
> 297/296 [======================] - 22s - loss: 0.0230 - val_loss: 0.0244
> 
> Epoch 13/15
> 297/296 [======================] - 22s - loss: 0.0219 - val_loss: 0.0240
> 
> Epoch 14/15
> 297/296 [======================]- 20s - loss: 0.0213 - val_loss: 0.0234
> 
> Epoch 15/15
> 297/296 [======================] - 20s - loss: 0.0201 - val_loss: 0.0247

## Result

The video recording of the car driving almost 2 laps autonomously can be watched on this repo (video.mp4). Here is a 60 seconds sample of the video:

![Video recording of vehicule driving](https://github.com/vincentbarrault/Behavioral-cloning/blob/master/Resource/video.gif?raw=true)
## Possible improvements

### Improving data collection

The quality of the dataset used for training the neural network is essential. This could be improved by:

- Collecting more data by driving more laps
- Driving from the sides during a "lap of recovery"
- Use Joystick instead of mouse to have a better precision (speed, steering angle)
- Recollecting datas on one specific part of the track where the model was not performing well
- Collecting datas on another track

 ### Resizing dataset images

The size of the images is 160x320 pixels. It is quite big and reducing the size of the images would make the CNN a lot faster to train, allowing to train our neural network on a bigger dataset. This would also bring less detailled images and affect the quality of the neural network output. A compromise between training speed and image size/quality should be found in this case.

### Adding more layers or modifying the existing layers to CNN

The CNN could be more complex but it would increase the time needed to train the neural network. This could be needed for harder tracks.
