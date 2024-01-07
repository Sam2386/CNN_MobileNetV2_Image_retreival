CNN_MobileNetV2_Image_retreival
 This is a repository containing three tasks which helped me understand the working of CNNs.
Task 1
Convolutional Neural Network (CNN) for Handwritten Digits Classification
 This script demonstrates the implementation of a CNN for the classification of handwritten digits into three classes (0, 1, and 2). The code uses TensorFlow and Keras for model development and training.

Key Components
1. Data Preparation
The script assumes the existence of a dataset containing handwritten digits in the specified directory (`.../MyDrive/data_assign_2/hand_written_digits`).
Images are loaded and augmented using the `ImageDataGenerator` from TensorFlow.keras.preprocessing.

2. Data Augmentation
Augmentation is performed to increase the diversity of the training dataset and improve the model's generalization.
Common augmentation techniques include rotation, zooming, width and height shifting, and shearing.

3. CNN Model Architecture
The CNN model is defined using the Sequential API of TensorFlow.keras.
It consists of convolutional layers with max-pooling for feature extraction, followed by fully connected layers for classification.
The final layer has three neurons with softmax activation, representing the three classes (0,1,2).

4. Model Compilation
The model is compiled with the Adam optimizer and categorical crossentropy loss, suitable for multi-class classification problems.
The chosen metric for monitoring during training is accuracy.

5. Data Splitting and Training
The dataset is split into training and validation sets using the `train_test_split` function from scikit-learn.
The model is trained on the training set for 10 epochs, with validation performed on a separate validation set.

6. Model Evaluation
After training, the model is evaluated on the validation set.
The script prints the validation accuracy, providing insight into the model's performance.

Task 2
Fine-Tuning Pre-trained MobileNetV2 for Handwritten Digits Classification
 This script demonstrates the process of fine-tuning a pre-trained MobileNetV2 model on a dataset of handwritten digits. The goal is to classify the digits into three classes: 0, 1, and 2. The code uses TensorFlow and Keras for model development and training.

Key Components
1. Data Preparation and Augmentation
The script assumes the existence of a dataset containing handwritten digits in the specified directory (`.../MyDrive/data_assign_2/hand_written_digits`).
Images are loaded and augmented using the `ImageDataGenerator` from TensorFlow.keras.preprocessing.
Augmentation includes rescaling, rotation, zooming, width and height shifting, and shearing.

2. Data Splitting and Training Set-Up
The dataset is split into training and validation sets using the `train_test_split` function from scikit-learn.
Image size and batch size are defined (`img_size = (100, 100)` and `batch_size = 32`).
A random seed is set for reproducibility.

3. Loading Pre-trained MobileNetV2 Model
The MobileNetV2 model is loaded with pre-trained weights from 'imagenet' but excluding the top classification layer.
The pre-trained layers are frozen to retain the learned features.

4. Custom Model Architecture
Custom classification layers are added on top of the MobileNetV2 base.
These include a global average pooling layer, a dense layer with ReLU activation, and a final dense layer with softmax activation for classification into three classes.

5. Model Compilation and Training
The model is compiled with the Adam optimizer and categorical crossentropy loss, suitable for multi-class classification problems.
The script then trains the model on the training set for 10 epochs, with validation performed on a separate validation set.

6. Model Evaluation
After training, the model is evaluated on the validation set.
The script prints the validation accuracy, providing insight into the model's performance.

Task 3
Image Retrieval with VGG16
 This Python script demonstrates image retrieval using a pre-trained VGG16 model. The goal is to select a query image from a set of query images and retrieve the top N (in this case N=4 so 4 similar images are neing displayed) similar images from a database. The similarity is measured using the Euclidean distance between the feature vectors extracted from the VGG16 model.

Key Components
1. Loading and Preprocessing Images
Images are loaded from the specified folders (`query_images` and `images_retreival_local_database`).
The VGG16 model requires images to be preprocessed, so each image is resized to (224x224) pixels and pixel values are adjusted using the `preprocess_input` function.

2. Feature Extraction with VGG16
The VGG16 model is loaded with pre-trained weights from ImageNet.
The fully connected layer 'fc2' is extracted as a feature vector for each image using the `extract_features` function.

3. Similar Image Retrieval
Euclidean distances are computed between the feature vector of the query image and all feature vectors in the database.
The top N images with the smallest distances are selected as similar images using the `find_similar_images` function.

4. Displaying Results
In the first cell of the notebook CNN_MobileNetV2_Image_Retreival, Results include printing the query image and the filenames of the top N similar images.
In the later two cells of Task 3, Images are displayed using Matplotlib, showcasing the query image and similar images side by side.
In the last two cells of Task 3, Images are displayed using Matplotlib, showcasing the query image in first row and similar images in the second row.
