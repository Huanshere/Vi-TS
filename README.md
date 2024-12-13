# 基于计算机视觉的室内热环境控制系统
树莓派 + 热成像摄像头 + 摄像头

## 快速开始

### 前置准备

1. 在树莓派上启用 I2C 接口:
   ```bash
   sudo raspi-config
   ```
   进入 `Interfacing Options` > `I2C` > 选择 `Yes` 启用

安装环境 并 启动:
```bash
bash run.sh
```
该脚本会安装 python, pip, vscode, 并 安装依赖

## 开发要求

使用 [rich](https://github.com/Textualize/rich) 打印规范:

- 🌡️ 温度数据: `rprint("[red]Temperature: 25.6°C[/red]")`
- 💧 湿度数据: `rprint("[blue]Humidity: 65.2%[/blue]")`
- 🎯 检测结果: `rprint("[green]Detection: Person found[/green]")`
- ⚠️ 警告信息: `rprint("[yellow]Warning: High temperature[/yellow]")`
- ❌ 错误信息: `rprint("[bold red]Error: Camera not found[/bold red]")`
- ℹ️ 普通信息: `rprint("[white]Info: System started[/white]")`
- ✅ 成功信息: `rprint("[green]Success: Connected[/green]")`

## 项目结构

```
Vi-TS/
├── ai/                        # AI 模型和分析相关
│   ├── analyze_video_302.py  # 使用 302.ai API 分析视频和图片
│   ├── prompts.py            # AI 模型提示词管理
│   └── _analyze_qwen.py      # 通义千问视觉模型分析（备用）
│
├── configs/                   # 配置文件
│   └── face_detect_setting.py # 人脸检测和温度分析配置
│
├── processors/                # 数据处理模块
│   ├── rgb_processor.py      # RGB 摄像头数据处理
│   ├── sensor_processor.py   # SHT35 温湿度传感器数据处理
│   └── thermal_processor.py  # 热成像摄像头数据处理
│
├── utils/                     # 工具函数
│   ├── check_cam.py          # 摄像头检测工具
│   └── rgb_cam_utils.py      # RGB 摄像头工具函数
│
└── run.sh                    # 项目启动脚本
```

每个模块的主要功能：

- **ai/**: 负责视频分析、图像理解等 AI 相关功能
- **configs/**: 存放各类配置文件，如人脸检测参数等
- **processors/**: 处理各类传感器数据，包括视频流、温度数据等
- **utils/**: 提供各类工具函数，如摄像头初始化等

详见 [技术文档](TECH.md)

## Roadmap

### 高优先级
- [x] 热成像面部节点温度追踪
- [-] AI 视觉分析
  - [ ] 视频流 YOLO 人体检测
  - [ ] 环境设备检测（空调、窗户等）
  - [ ] VLLM 分析（低帧率触发 + 高帧率详细分析）
- [ ] 数据管理
  - [ ] 按时间序列记录数据（JSON 格式）
- [ ] 在线学习模块
  - [ ] 训练模型
  - [ ] 推测人员热舒适度
- [ ] Streamlit 实时数据可视化

### 低优先级
- [ ] 云台摄像头人脸跟踪
- [ ] 人脸识别
- [ ] 人脸隐私保护（打码处理）