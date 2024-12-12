#!/bin/bash

# 检查并安装 Python 3.11 (Debian 12 默认版本)
if ! command -v python3 &> /dev/null; then
    echo "Installing Python 3.11..."
    sudo apt update
    sudo apt install -y python3 python3-venv python3-distutils
fi

# 检查 VS Code 是否已安装
if ! command -v code &> /dev/null; then
    echo "Installing Visual Studio Code..."
    wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
    sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
    sudo sh -c 'echo "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main" > /etc/apt/sources.list.d/vscode.list'
    sudo apt install -y apt-transport-https
    sudo apt update
    sudo apt install -y code
fi

# 检查虚拟环境是否存在
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate

# 更新 pip 并安装依赖
echo "Updating pip and installing requirements..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 检查人脸特征点检测模型是否存在
if [ ! -f "face_landmarker.task" ]; then
    echo "Downloading face landmark model..."
    wget -O face_landmarker.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
fi

# 运行程序
python thermal_face.py