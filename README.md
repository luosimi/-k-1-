# 📸 本地相似图片检测系统 (Local Similar Image Search)

## 📖 项目简介

这是一个高性能的本地化图片检索工具，旨在利用最前沿的深度学习技术，为用户提供精准、隐私安全的“以图搜图”体验。系统完全离线运行，无需上传数据，完美保障用户隐私。

本项目基于 **OpenCLIP ViT-H-14** 模型，能够提取图片的高维语义特征，不仅能识别物体，还能理解构图与风格。配合 **GPU 加速** 和 **批量处理** 机制，即便是数万张图片的本地图库也能实现秒级索引与检索。

## ✨ 核心功能

- **🚀 深度视觉理解**：集成 ViT-H-14 大模型，提供SOTA级别的特征提取能力。
- **⚡ 硬件加速引擎**：
  - 自动检测 **NVIDIA GPU (CUDA)**。
  - 支持 **Batch Processing (批量处理)**，大幅提升索引速度。
  - 实时显示运行设备（CPU/GPU）及显存状态。
- **🔍 毫秒级检索**：基于余弦相似度的高效向量检索算法。
- **💻 现代化 Web 界面**：
  - 简洁美观的 UI（Tailwind CSS）。
  - 支持原生文件夹选择（无需手动输入路径）。
  - 实时进度条显示索引构建状态。
- **🔒 隐私安全**：所有计算和数据存储均在本地完成，无需联网。

## 🛠️ 技术栈

- **核心语言**: Python 3.8+
- **深度学习**: PyTorch, OpenCLIP
- **Web 框架**: Flask
- **前端技术**: HTML5, JavaScript, Tailwind CSS
- **图像处理**: Pillow (PIL), NumPy

## 📦 安装与运行

### 1. 环境准备

确保您的电脑已安装 Python 3.8 或更高版本。
推荐安装 [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 以获得最佳性能（如果有 NVIDIA 显卡）。

### 2. 安装依赖

在项目根目录下打开终端，运行：

```bash
pip install -r requirements.txt
```

*注意：`torch` 和 `torchvision` 可能需要根据您的 CUDA 版本单独安装，详见 [PyTorch 官网](https://pytorch.org/get-started/locally/)。*

### 3. 启动应用

运行以下命令启动服务器：

```bash
python app.py
```

### 4. 使用说明

1.  浏览器访问终端显示的地址（通常为 `http://127.0.0.1:5000`）。
2.  点击 **"📂 Browse"** 按钮选择您想要建立索引的本地图片文件夹。
3.  点击 **"Start Sync"** 开始构建索引。
    - 界面会显示当前运行设备（GPU/CPU）。
    - 进度条会实时更新。
4.  索引完成后，上传一张图片或输入图片路径进行搜索。
5.  系统将返回图库中最相似的图片结果。

## 📂 项目结构

```
.
├── core/
│   ├── feature_extractor.py  # 特征提取逻辑
│   ├── indexer.py           # 索引构建与 DataLoader 批量处理
│   ├── model_loader.py      # 模型加载与设备自动检测
│   └── search_engine.py     # 向量检索实现
├── templates/
│   └── index.html           # 前端界面
├── utils/
│   ├── file_utils.py        # 文件扫描工具
│   └── math_utils.py        # 向量计算工具
├── app.py                   # Flask 主程序
├── config.py                # 配置文件
└── requirements.txt         # 项目依赖
```

## 📝 注意事项

- **首次运行**: 首次启动时会自动下载 ViT-H-14 模型权重（约 2.5GB），请保持网络连接或手动放置权重文件。
- **性能提示**: 
  - **GPU 模式**: 推荐使用 6GB 以上显存的显卡，速度极快。
  - **CPU 模式**: 如果没有检测到 GPU，会自动回退到 CPU，速度会较慢，请耐心等待。

---
*Created with ❤️ by Trae AI Assistant*
