### Disclaimer

The official Faster R-CNN code (written in MATLAB) is available [here](https://github.com/ShaoqingRen/faster_rcnn).
If your goal is to reproduce the results in our NIPS 2015 paper, please use the [official code](https://github.com/ShaoqingRen/faster_rcnn).

This repository contains a Python *reimplementation* of the MATLAB code.
This Python implementation is built on a fork of [Fast R-CNN](https://github.com/rbgirshick/fast-rcnn).
There are slight differences between the two implementations.
In particular, this Python port
 - is ~10% slower at test-time, because some operations execute on the CPU in Python layers (e.g., 220ms / image vs. 200ms / image for VGG16)
 - gives similar, but not exactly the same, mAP as the MATLAB version
 - is *not compatible* with models trained using the MATLAB code due to the minor implementation differences
 - **includes approximate joint training** that is 1.5x faster than alternating optimization (for VGG16) -- see these [slides](https://www.dropbox.com/s/xtr4yd4i5e0vw8g/iccv15_tutorial_training_rbg.pdf?dl=0) for more information

# *Faster* R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

By Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun (Microsoft Research)

This Python implementation contains contributions from Sean Bell (Cornell) written during an MSR internship.

Please see the official [README.md](https://github.com/ShaoqingRen/faster_rcnn/blob/master/README.md) for more details.

Faster R-CNN was initially described in an [arXiv tech report](http://arxiv.org/abs/1506.01497) and was subsequently published in NIPS 2015.

### License

Faster R-CNN is released under the MIT License (refer to the LICENSE file for details).

### Citing Faster R-CNN

If you find Faster R-CNN useful in your research, please consider citing:

    @inproceedings{renNIPS15fasterrcnn,
        Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
        Title = {Faster {R-CNN}: Towards Real-Time Object Detection
                 with Region Proposal Networks},
        Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
        Year = {2015}
    }

### Contents
1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Basic installation](#installation-sufficient-for-the-demo)
4. [Demo](#demo)
5. [Beyond the demo: training and testing](#beyond-the-demo-installation-for-training-and-testing-models)
6. [Usage](#usage)

### Requirements: software

1. Windows 10 64-bit
2. Vs2015
3. Python 2.7 (Recommend use Anaconda)
4. OpenCV >= 3.1.0

### Requirements: hardware

1. Nvidia GTX 1060 6GB

### Installation (sufficient for the demo)


1. Build caffe-fast-rcnn
   cd $FRCN_ROOT/caffe-fast-rcnn/scripts

   modify build_win.cmd: 
   Line 8, 9, 74, 76
   ```
   if NOT DEFINED WITH_NINJA set WITH_NINJA=0
   if NOT DEFINED CPU_ONLY set CPU_ONLY=0
   ```
   Line 25, 29
   ```
   if !PYTHON_VERSION! EQU 2 (
       set CONDA_ROOT=C:\Program Files\Anaconda2
   )
   :: Set python 3.5 with conda as the default python
   if !PYTHON_VERSION! EQU 3 (
       set CONDA_ROOT=C:\Program Files\Anaconda2
   )
   ```
   Line 172 (cudnn path)
   ```
   -DCUDNN_ROOT=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0 ^
   ```
    
2. Build py-faster-rcnn
   SET VS90COMNTOOLS=%VS140COMNTOOLS%
   Add cl.exe to env path: C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin
   cd $FRCN_ROOT/lib
   python setup.py install
   python setup_cuda.py install
   pip install -r requirements.txt
3. Run Demo
   SET PYTHONPATH=$FRCN_ROOT/caffe-fast-rcnn/python

### Demo

*After successfully completing [basic installation](#installation-sufficient-for-the-demo)*, you'll be ready to run the demo.

To run the demo
```Shell
cd $FRCN_ROOT
./tools/demo.py
```
The demo performs detection using a VGG16 network trained for detection on PASCAL VOC 2007.

### Beyond the demo: installation for training and testing models
1. Download the training, validation, test data and VOCdevkit

	```Shell
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
	```

2. Extract all of these tars into one directory named `VOCdevkit`

	```Shell
	tar xvf VOCtrainval_06-Nov-2007.tar
	tar xvf VOCtest_06-Nov-2007.tar
	tar xvf VOCdevkit_08-Jun-2007.tar
	```

3. It should have this basic structure

	```Shell
  	$VOCdevkit/                           # development kit
  	$VOCdevkit/VOCcode/                   # VOC utility code
  	$VOCdevkit/VOC2007                    # image sets, annotations, etc.
  	# ... and several other directories ...
  	```

4. Create symlinks for the PASCAL VOC dataset

	```Shell
    cd $FRCN_ROOT/data
    ln -s $VOCdevkit VOCdevkit2007
    ```
    Using symlinks is a good idea because you will likely want to share the same PASCAL dataset installation between multiple projects.
5. [Optional] follow similar steps to get PASCAL VOC 2010 and 2012
6. [Optional] If you want to use COCO, please see some notes under `data/README.md`
7. Follow the next sections to download pre-trained ImageNet models

### Download pre-trained ImageNet models

Pre-trained ImageNet models can be downloaded for the three networks described in the paper: ZF and VGG16.

```Shell
cd $FRCN_ROOT
./data/scripts/fetch_imagenet_models.sh
```
VGG16 comes from the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo), but is provided here for your convenience.
ZF was trained at MSRA.

### Usage

To train and test a Faster R-CNN detector using the **alternating optimization** algorithm from our NIPS 2015 paper, use `experiments/scripts/faster_rcnn_alt_opt.sh`.
Output is written underneath `$FRCN_ROOT/output`.

```Shell
cd $FRCN_ROOT
./experiments/scripts/faster_rcnn_alt_opt.sh [GPU_ID] [NET] [--set ...]
# GPU_ID is the GPU you want to train on
# NET in {ZF, VGG_CNN_M_1024, VGG16} is the network arch to use
# --set ... allows you to specify fast_rcnn.config options, e.g.
#   --set EXP_DIR seed_rng1701 RNG_SEED 1701
```

("alt opt" refers to the alternating optimization training algorithm described in the NIPS paper.)

To train and test a Faster R-CNN detector using the **approximate joint training** method, use `experiments/scripts/faster_rcnn_end2end.sh`.
Output is written underneath `$FRCN_ROOT/output`.

```Shell
cd $FRCN_ROOT
./experiments/scripts/faster_rcnn_end2end.sh [GPU_ID] [NET] [--set ...]
# GPU_ID is the GPU you want to train on
# NET in {ZF, VGG_CNN_M_1024, VGG16} is the network arch to use
# --set ... allows you to specify fast_rcnn.config options, e.g.
#   --set EXP_DIR seed_rng1701 RNG_SEED 1701
```

This method trains the RPN module jointly with the Fast R-CNN network, rather than alternating between training the two. It results in faster (~ 1.5x speedup) training times and similar detection accuracy. See these [slides](https://www.dropbox.com/s/xtr4yd4i5e0vw8g/iccv15_tutorial_training_rbg.pdf?dl=0) for more details.

Artifacts generated by the scripts in `tools` are written in this directory.

Trained Fast R-CNN networks are saved under:

```
output/<experiment directory>/<dataset name>/
```

Test outputs are saved under:

```
output/<experiment directory>/<dataset name>/<network snapshot name>/
```
