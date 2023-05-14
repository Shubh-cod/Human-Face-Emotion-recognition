# Human-Face-Emotion-recognition
The FER2013 dataset, which contains 48x48 pixel grayscale images of faces labelled with seven emotion categories is used for facial expression recognition tasks.

We used PyTorch in this project to implement, train, and test various deep learning models. The dataset was divided into two parts: a training set and a test set, with data augmentation techniques like random cropping, rotation, and horizontal flipping applied to the training set to improve the model's generalisation capabilities. We tried out the following models:
Custom Model: A custom CNN architecture was implemented, consisting of four convolutional layers followed by max-pooling and dropout layers, and two fully connected layers.
ResNet18: A pre-trained ResNet18 model was fine-tuned for the facial expression recognition task.
MobileNetV2: A pre-trained MobileNetV2 model was fine-tuned for the facial expression recognition task.
SqueezeNet1_0: A pre-trained SqueezeNet1_0 model was fine-tuned for the facial expression recognition task.
ShuffleNetV2_x1_0: A pre-trained ShuffleNetV2_x1_0 model was fine-tuned for the facial expression recognition task.
