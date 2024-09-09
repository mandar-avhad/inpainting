#!/bin/bash

# Install Python packages from requirements.txt
pip install -r requirements.txt

# Install additional Python packages
pip install -r ./third_party/lama_requirements.txt
pip install gradio==4.29.0
pip install fastapi==0.112.4
# pip install gradio-client==1.3.0
pip install pydantic==2.9.0
pip install pydantic-core==2.23.2
pip install diffusers
pip install git+https://github.com/openai/CLIP.git

# Change to the target directory (relative path)
cd pretrained_models || { echo "Directory not found"; exit 1; }

# Download the specific file
wget -q 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
