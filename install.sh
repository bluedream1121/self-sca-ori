#!/bin/bash


if [ "$#" -ne 2 ]; then
    echo "ERROR! Illegal number of parameters. Usage: bash install.sh conda_install_path environment_name"
    exit 0
fi

conda_install_path=$1
conda_env_name=$2

echo ""
echo ""

source $conda_install_path/etc/profile.d/conda.sh
echo "****************** Creating conda environment ${conda_env_name} python=3.8.5 ******************"
conda create -y --name $conda_env_name python=3.8.5

echo ""
echo ""
echo "****************** Activating conda environment ${conda_env_name} ******************"
conda activate $conda_env_name

echo "****************** Installing pytorch with cuda11.1 ******************"
conda install -y pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

echo ""
echo ""
echo "****************** Installing matplotlib ******************"
conda install -y matplotlib

echo ""
echo ""
echo "****************** Installing opencv, scipy, tqdm, exifread ******************"
pip install opencv-python
pip install scipy
pip install argparse
pip install tqdm
pip install wandb

