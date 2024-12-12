#!/bin/bash

# 检查并安装 Python 3.10
if ! command -v python3.10 &> /dev/null; then
    echo "Installing Python 3.10..."
    sudo apt update
    sudo apt install -y python3.10 python3.10-venv python3.10-distutils
fi

# 安装 VS Code
echo "Installing Visual Studio Code..."
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
sudo sh -c 'echo "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main" > /etc/apt/sources.list.d/vscode.list'
sudo apt install -y apt-transport-https
sudo apt update
sudo apt install -y code

# 创建并激活虚拟环境
echo "Creating and activating virtual environment..."
python3.10 -m venv venv
source venv/bin/activate

# 更新 pip 并安装依赖
echo "Updating pip and installing requirements..."
python3.10 -m pip install --upgrade pip
python3.10 -m pip install -r requirements.txt

# 下载人脸特征点检测模型
echo "Downloading face landmark model..."
wget -O face_landmarker.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# 退出虚拟环境
deactivate

echo "Setup completed successfully!"

# 运行方式：bash setup.sh
