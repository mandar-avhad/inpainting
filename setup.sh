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
curl -LJO https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip
unzip big-lama.zip
rm big-lama.zip

# Change to the parent directory
cd ..

# Change to the clipseg directory
cd clipseg || { echo "Directory not found"; exit 1; }

# Download and unzip weights
wget https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download -O weights.zip
unzip -d weights -j weights.zip
rm weights.zip

# Change back to the parent directory
cd ..
