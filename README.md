# 3362AnimeCharacterGenerator

## Objective
We plan to train a model based on “StyleGAN2” implemented on a webpage that generates unique and customizable 2D anime characters to be used as references for artists. There will be a variety of continuously adjustable features for generating new characters, such as “glasses”, “hair length”, “smile”, etc. Figure 1 shows an example that we want to achieve for our final product yet differing in adjustable options.

## Target Application
Creating artworks like illustrations or comics often involves making a number of original characters from scratch. It is easy for artists to come up with the first few characters with diverse features, but when the number of characters increases it turns out to be more difficult. To avoid similarity and amplify diversity, we plan to build this model to generate unique characters with customizable features and hopefully provide to artists as their inspirations of brainstorming.

## Dataset
“Crypko Data”: https://www.kaggle.com/shilou/crypko-data  
We plan to use this dataset from Kaggle, consisting of about 71,300 face images.   
Since the images are not tagged according to their features, we need to preprocess the dataset. Due to the large number of images, obviously it is impossible to label them manually. Therefore, we plan to train several sub models to classify the images and create tags for each of them. For example, we will train and apply a sub model for emotion classification.

## Methodology
We will be training a GAN model, specifically the variation StyleGAN2.  
A Generative Adversarial Network (GAN) is able to generate new content with the same statistics as the training dataset. Figure 2 shows the framework of GAN, which consists of two sub networks, a generator and a discriminator. The former generates fake images while the latter evaluates and distinguishes them from real images. The weights of two sub networks are then updated based on the loss.  
StyleGAN2 is a GAN based network with a modified inner architecture which allows it to adjust different levels of features of the generated images from the “coarse” details (eg. glasses) to “finer” details (eg. hair color). The model will initially be trained at 4x4 resolution. After stabilizing, layers with higher resolution (8x8, 16x16, 32x32, 64x64, 128x128, 256x256, 512x512, 1024x1024) are gradually added to the model. This is to accelerate the training process and also make the model stable and robust (Rashad, 2020).

