# Hybrid-CNNs-for-multi-class-segmentation
![Python](https://img.shields.io/badge/python-v3.9-blue)
![Tensor](https://img.shields.io/badge/TensorFlow-V2.9.1-orange)
![Keras](https://img.shields.io/badge/Keras-V2.7-brightgreen)
![pandas](https://img.shields.io/badge/Pandas-V1.4.2-ff69b4)
![numpy](https://img.shields.io/badge/%E2%80%8ENumpy-V1.20.2-success)
![releasedate](https://img.shields.io/badge/release%20date-april%202022-red)
![Opensource](https://img.shields.io/badge/OpenSource-Yes!-6f42c1)


# Different  Hybrid U-Net base - Neural Networks architecture for categorical Semantic Segmentation from telecentric wide field reflected microscopic time-series images



This is the official repository of **Different U-Net - Neural Networks architecture for Semantic Segmentation from bright field microscopic time-series images** 


**Compression between Different U-Net - Neural Networks architecture for Semantic Segmentation** <br />

[Ali Ghaznavi<sup>∗</sup>](http://web.frov.jcu.cz/cs/o-fakulte/soucasti-fakulty/ustav-komplexnich-systemu-uks/labo-exp-komplex-systemu), 
[Renata Rychtáriková<sup>∗</sup>](http://web.frov.jcu.cz/cs/o-fakulte/soucasti-fakulty/ustav-komplexnich-systemu-uks/labo-exp-komplex-systemu), 
[Petr Císař<sup>∗</sup>](http://web.frov.jcu.cz/en/about-faculty/faculty-parts/institute-complex-systems/lab-signal-image-processing2),
[Mohammadmehdi Ziaei<sup>∗</sup>](http://web.frov.jcu.cz/en/about-faculty/faculty-parts/institute-complex-systems/lab-signal-image-processing2),
[Dalibor Štys<sup>∗</sup>](http://web.frov.jcu.cz/cs/kontakty-frov-ju/181-prof-rndr-dalibor-stys-csc) <br />
(* *indicates equal contribution*)

**For details, please refer to:**

**[[Paper](https://www.researchgate.net/publication/359437115_Cell_segmentation_from_telecentric_bright-field_transmitted_light_microscopic_images_using_a_Residual_Attention_U-Net_a_case_study_on_HeLa_line)]** 



## Abstract

Multi-class segmentation of unlabelled living cells in time-lapse light microscopy images is challenging due to the temporal behaviour and changes in cell life cycles and the complexity of images of this kind. The deep learning-based methods achieved promising outcomes and remarkable success in single- and multi-class medical and microscopy image segmentation. 
The main objective of this study is to develop a hybrid deep learning-based categorical segmentation and classification method for living HeLa cells in reflected light microscopy images.
%Method
Different hybrid convolution neural networks -- a simple U-Net, VGG19-U-Net, Inception-U-Net, and ResNet34-U-Net architectures -- were proposed and mutually compared to find the most suitable architecture for multi-class segmentation of our datasets.  

The inception module in the Inception-U-Net contained kernels with different sizes within the same layer to extract all feature descriptors. The series of residual blocks with the skip connections in each ResNet34-U-Net's level alleviated the gradient vanishing problem and improved the generalisation ability.
%Result
The m-IoU scores of multi-class segmentation for our datasets reached 0.7062, 0.7178, 0.7907, and 0.8067 for the simple U-Net, VGG19-U-Net, Inception-U-Net, and ResNet34-U-Net, respectively. For each class and the mean value across all classes, the most accurate multi-class semantic segmentation was achieved using the ResNet34-U-Net architecture (evaluated as the m-IoU and Dice metrics).
## Introduction

##### Hybrid deep-learning multi-class segmentation of HeLa cells in reflected light microscopy images ###### 

The data  achieved by reflected light microscope from living Hela cells in different time-laps experiments under the condition already have been described in manuscript and divided to train, test and validation sets.

The labeled data have been prepared manually to train with the deep learning based methods

The models have been trained based on four hybrid different CNN architecture (with the size of 512 * 512) to achieve the best segmentation result already reported in manuscript.



## (1) Dataset and pre-trained model

**The Data-Set is Available in below links:**

[To download Dataset you can use this link:] [Click Here](https://datadryad.org/stash/dataset/doi:10.5061/dryad.80gb5mksp)	"Microscopic data-set web directory include:       Training, Testing and Validation datasets are separately available in the linked repository."


## (2) Methodology and DNN Architectures

We use this Deep Neural Network architecture:
Modelling Bright Filed Dataset on U-Net Networks:


<p align="center"> <img src="method.png" width="95%"> </p>


**Important hyperparameters setup:**

 Image Size  = 512 * 512

 number of layer ; default = 5

 Activation function = Leaky ReLU

 epoch size; default = 100

 batch size; default = 8

 Early Stop = 15

 learning rate ; default = 10e -3

 Step per Epoch = 100

 dropout_rate = 0.05




<p align="center"> <img src="stucture.png" width="95%"> </p>

## (3) Usage

**To run the script please use this file on Google Colab or Jupyter Notebook:**

```python
U_Net_ATT_Unet+Res_Unet_Pub_V2.ipynb
```




## (4) Evaluation and metrics

**We uses evaluation Metrics for experimental results:**

Precision, Recall, Intersection over Union (IoU), Accuracy, Dice 

## (5) Citation

If you find our work useful in your research, please consider citing:

    @article{unknown,
    author = {Ghaznavi, Ali and Rychtarikova, Renata and Saberioon, Mehdi and Stys, Dalibor},
    year = {2022},
    month = {03},
    pages = {},
    doi = {https://doi.org/10.1016/j.compbiomed.2022.105805}
    title = {Cell segmentation from telecentric bright-field transmitted light microscopic images using a Residual Attention U-Net: a case study on HeLa line}
    }

## (6) Updates

* 22/03/2022: Adding the dataset information

* -----: Initial release.
