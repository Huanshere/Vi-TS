#!/bin/bash

# 添加用户输入提示
echo "Do you want to perform a full reinstall? (y/n)"
read answer

if [ "$answer" = "y" ]; then
    # 执行 git pull
    echo "Updating from git repository..."
    git pull

    # 检查并安装 Python 和必要的包
    if ! command -v python3 &> /dev/null; then
        echo "Installing Python..."
        sudo apt update
        sudo apt install -y python3 python3-distutils
    fi

    # 检查并安装 pip
    if ! command -v pip3 &> /dev/null; then
        echo "Installing pip..."
        sudo apt install -y python3-pip
    fi

    # 检查 VS Code 是否已安装
    if ! command -v code &> /dev/null; then
        echo "Installing Visual Studio Code..."
        wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
        sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
        sudo sh -c 'echo "deb [arch=amd64,arm64,armhf] http://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
        sudo apt install -y apt-transport-https
        sudo apt update
        sudo apt install -y code-insiders
    fi

    # 直接安装依赖
    echo "Installing requirements..."
    python3 -m pip install --break-system-packages -r requirements.txt

else
    echo "Skipping installation steps..."
fi

# 运行程序
python3 thermal_face.py