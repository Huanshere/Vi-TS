# 基于计算机视觉的室内热环境控制系统
Indoor Thermal Environment Control System Based on Computer Vision

## 快速开始

安装环境 并 启动:
```bash
bash onekey.sh
```

## Roadmap

### 高优先级
- [x] 热成像面部节点温度追踪
- [ ] AI 视觉分析
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