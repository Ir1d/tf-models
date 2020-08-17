# TF2 Implementation of "Learning-to-See-in-the-Dark"

<!-- > :memo: Add a badge for the ArXiv identifier of your paper (arXiv:YYMM.NNNNN) -->

<!-- [![Paper](http://img.shields.io/badge/Paper-arXiv.YYMM.NNNNN-B3181B?logo=arXiv)](https://arxiv.org/abs/...) -->

This repository is the unofficial implementation of the following paper.

* Paper title: [Learning-to-See-in-the-Dark](http://cchen156.github.io/paper/18CVPR_SID.pdf)

## Description

The official TF1 implementation could be found at [cchen156/Learning-to-See-in-the-Dark](https://github.com/cchen156/Learning-to-See-in-the-Dark).

## History

<!-- > :memo: Provide a changelog. -->

- 200817: add docs

## Maintainers

<!-- > :memo: Provide maintainer information.   -->

* Dejia Xu ([@ir1d](https://github.com/ir1d))

## Requirements

[![TensorFlow 2.2](https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)
[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)

The dependencies should be same as the official implementation.

To run the code, you need Scipy + Numpy + Rawpy installed.

## Results

PSNR on whole Sony dataset: 28.556911 (Original: 28.88, [cydonia999](https://github.com/cydonia999/Learning_to_See_in_the_Dark_PyTorch#psnrssim-results): 28.55)

Remove the misaligned pairs as [suggested](https://github.com/cchen156/Learning-to-See-in-the-Dark#dataset): 28.90165.

## Dataset

Please follow the [instructions by the authors](https://github.com/cchen156/Learning-to-See-in-the-Dark#dataset) to obtain the dataset.

## Training

```shell
python3 train_Sony.py
```

The trained models should be saved in `checkpoint/Sony` folder.

## Evaluation

```shell
python3 test_Sony.py
```

## License

MIT License

## Citation

If you use our code and dataset for research, please cite the authors' paper:

Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun, "Learning to See in the Dark", in CVPR, 2018.
