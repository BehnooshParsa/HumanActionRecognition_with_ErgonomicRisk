# Human Action Recognition and Ergonomic Risk Assessment 

This is a step by step implementation of the method we used for human action recognition and ergonomic risk assessement in ["Toward Ergonomic Risk Prediction via Segmentation of Indoor Object Manipulation Actions Using Spatiotemporal Convolutional Networks"](https://ieeexplore.ieee.org/document/8746140) paper that is accepted to the [CASE conference 2019](https://www.ieee-ras.org/component/rseventspro/event/1488-case-2019-international-conference-on-automation-science-and-engineering) and the IEEE Robotics and Automation Letters.

### Dataset

The UW-IOM dataset can be found [here](https://data.mendeley.com/datasets/xwzzkxtf9s/1).

The TUM Kitchen dataset can be found [here](https://ias.in.tum.de/dokuwiki/software/kitchen-activity-data). We relabeled the dataset, and labels can be dounloaded in this repository under the folder "Labels_TUM".

### Requirements

This code has been tested on a workstation running Windows 10 operating system, equipped with a 3.7GHz 8 Core Intel Xeon W-2145 CPU, GPU ZOTAC GeForce GTX 1080 Ti, and 64 GB RAM.

* TensorFlow, Keras (1.1.2+)

* Tested on Python 3.6

### File structure

If you creat a directory for this project and copy the code in the "Code" folder and the UW-IOM dataset in the "data" folder the rest of the required directories will be generated automatically. 

```
.\Code
.\UW_IOM_Dataset
```
### Steps
1- Preparing the data is described in "Preparing_the_data.ipynb"

2- The feature extraction phase is described in "VGG16.ipynb"

3- The Temporal Convolutional Network is described in "TCN_Main_GPU.ipynb"
## Acknowledgments

* The TCN code was built on the code by [Colin Lea](https://github.com/colincsl/TemporalConvolutionalNetworks) presented in the [Temporal Convolutional Networks for Action Segmentation and Detection](https://arxiv.org/abs/1611.05267).

## Citation
Please cite the following article if you found the code and UW-IOM dataset useful:

B. Parsa, E. U. Samani, R. Hendrix, C. Devine, S. M. Singh, S. Devasia, and A. G. Banerjee. Toward Ergonomic Risk Prediction via Segmentation of Indoor Object Manipulation Actions Using Spatiotemporal Convolutional Networks. IEEE Robotics and Automation Letters, To appear. [[Pre-print]](https://arxiv.org/abs/1902.05176)
