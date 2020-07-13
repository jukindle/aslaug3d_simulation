# Aslaug3D Simulation instructions

## Installation
Note that this repository uses python3. After cloning this repository, cd into it and execute

    mkdir -p data/saved_models

### Dependencies
Install the packages [gym](https://github.com/openai/gym) (0.14.0), [stable_baselines](https://github.com/hill-a/stable-baselines) (2.7.0), [numpy](https://github.com/numpy/numpy) (1.16.2), [scipy](https://github.com/scipy/scipy) (1.3.1), [pybullet](https://github.com/bulletphysics/bullet3) (2.5.6), [pyyaml](https://pypi.org/project/PyYAML/) (5.1.2) and [opencv](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html) (4.1.1). Note that the minimalistic installation with pip might be sufficient (opencv-python==4.1.1.26), but is not guaranteed.


## Usage

### Training
To train an agent for 90M steps with 16 worker in parallel stored to the folder `test_training`, use

    python3 train.py -s 90e6 -v v0 -p policies.aslaug_policy_v0.AslaugPolicy -f test_training -n 16

### Running
To run a trained agent from folder `test_training` at episode 60.5M, use

    python3 run.py -v v0 -f test_training:60.5M

### Playing with parameters
Every time a training session is started, the Hyperparameters for the training and the environment are loaded from the file `params.yaml`. Therefore, it is recommended to parametrize new features added to the environments to allow for maximum flexibility and fast adjustment of parameters.

### Convertion of an arbituary robot xacro file to pybullet URDF
In order to eliminate the need of a functioning ROS installation, we created a script which converts a xacro file to URDF without ROS package references but relative paths. Note that for the conversion, ROS must be installed and the workspace in which the robot description is located must be sourced. Also, note that ROS uses python2, so this file should be executed with python2 as well.

In order to convert a xacro file to a reference-less URDF with all dependencies copied to a subfolder in the `urdf` folder, use

    python xacro_to_pybullet_urdf.py '/path/to/xacro/file'

This will create a new folder in `urdf` with the same name as the xacro file, filled with a urdf file which describes the robot and a subfolder `meshes` with all required files (.stl, .dae etc.).

## Citing

Please cite the [following paper](https://arxiv.org/abs/2003.02637) when using this repository for your paper:

```bibtex
@article{kindle2020rlwbc,
  title={Whole-Body Control of a Mobile Manipulator using End-to-End Reinforcement Learning},
  author={Kindle, Julien and Furrer, Fadri and Novkovic, Tonci and Chung, Jen Jen and Siegwart, Roland and Nieto, Juan},
  journal={arXiv preprint arXiv:2003.02637},
  year={2020}
}
```
