# Single-Shot-HDR-Imaging-with-MSCNN

We provide MatCaffe and PyTorch versions and data for our paper.

# Prerequisites:
+ Python 3.8
+ Pytorch 1.8.1
+ Matlab 2020a

# MatCaffe version
For testing:
+ Running "test_main.m"

> Note: before running "test_main.m", installing caffe (MatCaffe) and setting path for HDR metrics.

Requirements:
+ Download network weight from: https://drive.google.com/file/d/1AZvN7zor7pjujbly4D0GMURejXJzO7m0/view?usp=sharing
+ Download HDR-VDP metric from: https://drive.google.com/file/d/18nvNi4NEwBDPiIJEM-DMwChRfViID0gy/view?usp=sharing. And extract to the IQAs folder.
+ Download caffe library from: https://drive.google.com/file/d/1XNSGGAN0pTaY1kdxYxnQVSHMrLD2uT5p/view?usp=sharing

# PyTorch version
For testing:
+ Running "Generating_test_images.py" to produce test outputs
+ Then running "MAIN_TEST.m" to evaluate with HDR metrics

Requirement:
+ Download network weight from: https://drive.google.com/file/d/1pABE5JPl9HfARtxwA-pBW3uJ6E4YMFuF/view?usp=sharing

> Note: setting path for HDR metrics is the same with MatCaffe version

# Training Data
In preparing

Citing Single-Shot HDR Imaging with MSCNN
-------------
If you find our work useful in your research, please consider citing:

    @ARTICLE{Vien_2021,
      author={Vien, An Gia and Lee, Chul},
      journal={IEEE Access}, 
      title={Single-Shot High Dynamic Range Imaging via Multiscale Convolutional Neural Network}, 
      year={2021},
      volume={9},
      number={},
      pages={70369-70381},
      doi={10.1109/ACCESS.2021.3078457}
     }

