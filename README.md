# Single-Shot-HDR-Imaging-with-MSCNN

We provide MatCaffe and PyTorch versions and data for our paper:

An Gia Vien and Chul Lee, “Single-shot high dynamic range imaging via multiscale convolutional neural network,” accepted for publication in IEEE Access, May 2021.

# Prerequisites:
+ Python 3.xxx
+ Matlab 2020a

# MatCaffe version
Including:
+ Network details (prototxt file)
+ Testing file (test_main.m)
+ Testing images ("Test_imgs" folder including: input and ground-truth)
+ HDR metrics (HDR-VDP, pu-PSNR, log-PSNR)

> Note: before running "test_main.m", installing caffe (MatCaffe) and setting path for HDR metrics.

Requirements:
+ Download network weights from: https://drive.google.com/file/d/1lVmEQ-WZqjUq8xQSh4vqqvEseiCbsYLY/view?usp=sharing 
+ Download HDR-VDP metric from: https://drive.google.com/file/d/18nvNi4NEwBDPiIJEM-DMwChRfViID0gy/view?usp=sharing. And extract to the IQAs folder.
+ Download caffe library from: https://drive.google.com/file/d/1XNSGGAN0pTaY1kdxYxnQVSHMrLD2uT5p/view?usp=sharing

# PyTorch version
+ Running "Generating_test_images.py" to produce test outputs
+ Then running "MAIN_TEST.m" to evaluate with HDR metrics

> Note: setting path for HDR metrics is the same with MatCaffe version

# Data (Training set & Testing set)
In preparing
