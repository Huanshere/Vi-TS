# Gemini 视频分析示例

`analyze_video.py` 是一个使用 Gemini Pro Vision 模型进行视频分析的示例代码。

## 使用说明与成本分析

- 支持分辨率: 768x768 像素及以下
- 单张图片消耗: 258 tokens
- 详细资源消耗说明: [Vertex AI 输入文件要求](https://firebase.google.com/docs/vertex-ai/input-file-requirements?hl=zh-cn)

### 成本计算

当前版本:
- 每10秒消耗约 $0.000438 ？似乎现在的10s输入token显示才 29？
- 每小时成本约 ¥1
- 默认采样率为 1fps (每秒1帧)

优化建议:
- 可自行调整采样率至 10fps 以提升动作识别精度
- 提升后每小时成本约 ¥10
- 期待 Gemini 2.0 Flash 版本发布，将大幅降低使用成本
