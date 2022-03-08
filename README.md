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

## Weekly log 1 (2022/03/01 - 2022/03/08)
#### Objectives
- [x] Test if the pretrained CNN model works well
- [x] Estimate the time for the whole labelling process
- [ ] Label the dataset (~70000 images)
#### What has been achieved
It is confirmed that iv2 works well. Labelling 100 images takes about 40 seconds. The whole labelling process is estimated to be 40 * 700 / 60 / 60 = 7.78 hours, which turns out to be feasible.
#### Problems & Solutions
The variety of the features seems to be small. For 100 random samples in the dataset, features labelled are counted as follow:
- {'1girl': 100, 'solo': 99, 'blue eyes': 28, 'face': 25, 'open mouth': 19, 'pink hair': 12, 'blonde hair': 11, 'green eyes': 8, 'brown hair': 7, 'short hair': 6, 'red eyes': 6, 'smile': 5, 'brown eyes': 4, 'yellow eyes': 4, 'white hair': 4, 'blush': 4, 'long hair': 3, 'purple hair': 2, 'blue hair': 2, 'black hair': 2, 'aqua eyes': 2, 'monochrome': 2, 'purple eyes': 1, 'green hair': 1, 'aqua hair': 1, 'pink eyes': 1, 'hair ornament': 1, 'hat': 1, 'bunny ears': 1, 'school uniform': 1, 'one eye closed': 1, 'red hair': 1, 'traditional media': 1, 'sketch': 1}
- {'shigure (kantai collection)': 2, 'kaname madoka': 1, 'shana': 1, 'tenshi (angel beats!)': 1, 'saber': 1, 'nagato yuki': 1}
#### Plan for next week
Complete the labeling and start experimenting on training with StyleGAN2.
