# GESR: A Geometric Evolution Model for Symbolic Regression  
  
<p align="center">
  <a href="https://github.com/MZT-srcount/GESR?tab=readme-ov-file#abstract">Abstract</a> |
  <a href="https://github.com/MZT-srcount/GESR?tab=readme-ov-file#setup">Setup</a> |
  <a href="https://github.com/MZT-srcount/GESR?tab=readme-ov-file#run-the-model">Run the model</a> |
  <a href="https://github.com/MZT-srcount/GESR?tab=readme-ov-file#dependencies">Dependencies</a> |
  <a href="https://github.com/MZT-srcount/GESR?tab=readme-ov-file#examples">Examples</a> |
</p>

![image](/result/strogatz_bacres1.gif)

This repository is the official implementation of "GESR"




## Abstract

In this work, we propose a Geometric Evolution Symbolic Regression(GESR) algorithm. Three key modules are presented in GESR to enhance the approximation: (1) a new semantic gradient concept, proposed from the observation of inaccurate approximation results within semantic backpropagation, to assist the exploration and improve the accuracy of semantics approximation; (2) a new geometric semantic search operator, tailored for efficiently approximating the target formula directly in the sparse topological space, to obtain more accurate and interpretable solutions under strict program size constraints; (3) the Levenberg-Marquardt algorithm with L1 regularization, used for the adjustment of expression structures and the optimization of global subtree weights to assist the proposed geometric semantic search operator.


## Setup

### Prerequisites

- Supported Operation Systems: ``Linux``
- CUDA driver version ``>11.8``

### Install dependencies

Using conda and the environment.yml file:

- Run `conda env create -n GESR -f environment.yml`.

## Run the model

To launch a model training, a command without any additional arguments can be used, which will launch a model training for livermore datasets:

```bash
python ./autorun.py
```

You can also use additional arguments to train the specified datasets or specify the save path, like:

```bash
python ./autorun.py -train_dataset ./Dataset/easy/train/ -test_dataset ./Dataset/easy/test/ -save_file ./result/Feynman_easy_result.txt -seed 1111
```

## Dependencies

- python3
- numpy
- pycuda

## Examples

When you run the commands before in a terminal, you will get the following prints if it works successfully:

![image](/result/run_show.svg)

## License
The majority of this repository is released under the MIT license as found in the LICENSE file.