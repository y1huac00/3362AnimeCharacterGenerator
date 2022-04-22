# 3362AnimeCharacterGenerator

![Preview](https://github.com/y1huac00/3362AnimeCharacterGenerator/blob/main/preview.gif)

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

On Windows, you need to mannually add the path of your MSVC to <b>./stylegan2/dnnlib/tflib/custom_ops.py</b> as the following example:
```
compiler_bindir_search_path = [
    'C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Tools/MSVC/14.16.27023/bin/Hostx64/x64',
    'C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64',
    'C:/Program Files (x86)/Microsoft Visual Studio 14.0/vc/bin',
]
```

## How to play?
1. Clone the whole git repository including the sub-module stylegan2, download the stylegan2 model in 'Release' section
2. Prepare an environment according to Requirement
3. Lauch your Jupyter notebook and open anime_character_generator.ipynb
4. Run all codeblocks and pip install any packages that is required
5. Start to play!!
