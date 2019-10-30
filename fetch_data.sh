#!/bin/bash

# Script that fetches all necessary data for eval

# Model constants etc.
wget http://visiondata.cis.upenn.edu/texturepose/data.tar.gz && tar -xvf data.tar.gz && rm data.tar.gz
# List of preprocessed .npz files for each dataset
wget http://visiondata.cis.upenn.edu/texturepose/dataset_extras.tar.gz && tar -xvf dataset_extras.tar.gz --directory data && rm -r dataset_extras.tar.gz
# Pretrained checkpoint
wget http://visiondata.cis.upenn.edu/texturepose/texturepose_checkpoint.pth --directory-prefix=data

# Get smplx package
svn checkout https://github.com/vchoutas/smplx/trunk/smplx
