# 3362AnimeCharacterGenerator

![Preview](https://github.com/y1huac00/3362AnimeCharacterGenerator/blob/main/image.png)

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

## Requirement (from stylegan2 official)
* Both Linux and Windows are supported. Linux is recommended for performance and compatibility reasons.
* 64-bit Python 3.6 installation. We recommend Anaconda3 with numpy 1.14.3 or newer.
* We recommend TensorFlow 1.14, which we used for all experiments in the paper, but TensorFlow 1.15 is also supported on Linux. TensorFlow 2.x is not supported.
* On Windows you need to use TensorFlow 1.14, as the standard 1.15 installation does not include necessary C++ headers.
* One or more high-end NVIDIA GPUs, NVIDIA drivers, CUDA 10.0 toolkit and cuDNN 7.5. To reproduce the results reported in the paper, you need an NVIDIA GPU with at least 16 GB of DRAM.
* Docker users: use the [provided Dockerfile](./Dockerfile) to build an image with the required library dependencies.

StyleGAN2 relies on custom TensorFlow ops that are compiled on the fly using [NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html). To test that your NVCC installation is working correctly, run:

```.bash
nvcc test_nvcc.cu -o test_nvcc -run
| CPU says hello.
| GPU says hello.
```

On Windows, the compilation requires Microsoft Visual Studio to be in `PATH`. We recommend installing [Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/) and adding into `PATH` using `"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"`.


## How to play?
1. Clone the whole git repository including the sub-module stylegan2, download the stylegan2 model in 'Release' section
2. Prepare an environment according to Requirement
3. Lauch your Jupyter notebook and open anime_character_generator.ipynb
4. Run all codeblocks and pip install any packages that is required
5. Start to play!!
