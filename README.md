# ANTS
Code for paper "Deep Co-Attention Network for Multi-View Subspace Learning"

## Environment
* Python 3.7
* Pytorch
* Tensorflow (For downloading mnist dataset)

## Demo
1. Download dataset from http://www.vision.caltech.edu/visipedia/CUB-200.html
2. Download the mask-rcnn from https://github.com/matterport/Mask_RCNN, and install the required package.
2. Run the following command to generate two views as well as the image segments: 
```bash
python mask_rcnn.py
```
3. Run the following command to get the hidden representation for the final training: 
```bash
python pre-train_vgg.py 
```
4. Run the following command to train the model: 
```bash
python main.py -g 0 -e 150 -hid 300 -d birds
```
