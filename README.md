# StegaVAR: 基于隐写域分析的隐私保护视频行为识别

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-ee4c2c.svg)](https://pytorch.org/)

> 论文"StegaVAR: Privacy-Preserving Video Action Recognition via Steganographic Domain Analysis"的官方代码库

## 📖 简介

StegaVAR 是一个创新的隐私保护视频行为识别框架，通过隐写术技术将视频数据隐藏在载体视频中，从而在保护用户隐私的同时实现准确的行为识别。本项目结合了深度学习、隐写术和视频分析技术。

### 主要特点

- 🔒 **隐私保护**：通过隐写术保护敏感视频内容
- 🎯 **准确识别**：支持高精度的视频行为识别任务
- 🔄 **多模型支持**：集成多种隐写网络（LF-VSN, HiNet, WengNet, HiDDeN）
- 📊 **多数据集**：支持 UCF101、HMDB51、VisPR 等主流数据集
- ⚡ **灵活训练**：支持多种模型架构和训练策略

## 🏗️ 项目结构

```
StegaVAR/
├── main.py                 # 主程序入口
├── src/
│   ├── train_action.py     # 行为识别训练脚本
│   ├── train_privacy.py    # 隐私保护训练脚本
│   ├── hide_vid.py         # 视频隐藏功能
│   ├── model/              # 模型定义
│   │   ├── resnet2d.py
│   │   ├── resnet3d.py
│   │   ├── resnet3d_pro.py
│   │   ├── layers.py
│   │   └── get_model.py
│   ├── dataloader/         # 数据加载器
│   │   ├── ucf101.py
│   │   ├── hmdb51.py
│   │   ├── vispr.py
│   │   └── get_data.py
│   └── utils/              # 工具函数
│       └── video.py
└── README.md
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.8+
- CUDA 10.2+ (推荐使用 GPU)
- 其他依赖库：numpy, scikit-learn 等

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/clxhsa/StegaVAR.git
cd StegaVAR

# 安装依赖
pip install torch torchvision numpy scikit-learn
# 根据项目需要安装其他依赖
```

### 数据集准备

项目支持以下数据集：

- **UCF101**：包含 101 个行为类别的视频数据集
- **HMDB51**：包含 51 个行为类别的视频数据集
- **VisPR**：隐私保护相关的视频数据集

请按照各数据集的官方说明下载并组织数据。

## 💻 使用方法

### 基础训练（无隐写）

在 UCF101 数据集上训练行为识别模型：

```bash
python main.py \
  --run_id nohide_r3d_ta \
  --task har \
  --model r3dpro_ta \
  --batch_size 32 \
  --num_workers 8 \
  --pin_memory \
  --train_data ucf101 \
  --val_data ucf101
```

在 HMDB51 数据集上训练：

```bash
python main.py \
  --run_id hmdb51_nohide \
  --task har \
  --train_data hmdb51 \
  --val_data hmdb51 \
  --model r3dpro_ta \
  --batch_size 32 \
  --num_workers 8 \
  --pin_memory \
  --learning_rate 1e-4 \
  --alpha 0.2 \
  --beta 0.3 \
  --theta 0.0
```

### 隐写训练（启用隐私保护）

使用 LF-VSN 隐写模型：

```bash
python main.py \
  --hide \
  --run_id ucf101_r3dpro_ta \
  --task har \
  --model r3dpro_ta \
  --hide_model lfvsn \
  --batch_size 64 \
  --num_workers 8 \
  --pin_memory \
  --learning_rate 1e-4 \
  --alpha 0.2 \
  --beta 0.3 \
  --theta 0.2
```

使用不同的隐写网络：

```bash
# HiNet
python main.py --hide --hide_model hinet --run_id ucf101_hinet --task har --model r3dpro_ta

# WengNet
python main.py --hide --hide_model wengnet --run_id ucf101_wengnet --task har --model r3dpro_ta

# HiDDeN
python main.py --hide --hide_model hidden --run_id ucf101_hidden --task har --model r3dpro_ta
```

### 隐私保护任务训练

```bash
python main.py \
  --hide \
  --run_id ucf101_privacy \
  --task pri \
  --model r50 \
  --batch_size 64 \
  --num_workers 4 \
  --pin_memory \
  --num_epochs 100 \
  --train_data ucf101 \
  --val_data ucf101 \
  --val_int 1
```

### 模型微调

从预训练模型继续训练：

```bash
python main.py \
  --hide \
  --train_data hmdb51 \
  --val_data hmdb51 \
  --run_id hmdb51_finetune \
  --task har \
  --model r3dpro_ta \
  --batch_size 32 \
  --pin_memory \
  --alpha 0.2 \
  --beta 0.3 \
  --theta 0.2 \
  --saved_model ckpt/path/to/pretrained_model.pth
```

## ⚙️ 主要参数说明

### 任务参数

- `--task`：任务类型
  - `har`：行为识别 (Human Action Recognition)
  - `pri`：隐私保护 (Privacy)
- `--hide`：是否启用隐写模式
- `--run_id`：实验运行标识符

### 模型参数

- `--model`：主模型架构
  - `r3d18`：3D ResNet-18
  - `4r3d`：4-stream 3D ResNet
  - `r3dpro`：改进的 3D ResNet
  - `r3dpro_ta`：带时序增强的 3D ResNet（推荐）
  - `r50`：ResNet-50（用于隐私任务）
  - `vit`：Vision Transformer
  
- `--hide_model`：隐写模型
  - `lfvsn`：LF-VSN（默认）
  - `hinet`：HiNet
  - `wengnet`：WengNet
  - `hidden`：HiDDeN

### 数据集参数

- `--train_data` / `--val_data`：数据集选择（ucf101, hmdb51, vispr1, vispr2）
- `--num_classes`：类别数量（UCF101: 101, HMDB51: 51）
- `--num_frames`：每个视频片段的帧数（默认：16）
- `--reso_h` / `--reso_w`：输入分辨率（默认：224x224）

### 训练参数

- `--batch_size`：批大小（默认：64）
- `--learning_rate`：学习率（默认：1e-4）
- `--num_epochs`：训练轮数（默认：200）
- `--num_workers`：数据加载线程数（默认：4）

### 增强参数

- `--alpha`：空间提升参数（默认：0.0）
- `--beta`：时序提升参数（默认：0.0）
- `--theta`：CBDA（Cross-Band Domain Analysis）参数（默认：0.2）

## 📊 模型架构

### 支持的模型

1. **3D ResNet 系列**
   - R3D-18：基础 3D ResNet
   - R3DPro：改进版带有时空注意力机制
   - R3DPro-TA：带时序增强的版本（最佳性能）

2. **隐写网络**
   - **LF-VSN**：轻量级频域视频隐写网络
   - **HiNet**：层次化可逆神经网络
   - **WengNet**：Weng 等人提出的隐写网络
   - **HiDDeN**：基于深度学习的数据隐藏网络

3. **2D 模型**
   - ResNet-50：用于隐私保护任务

## 📈 实验结果

模型在 UCF101 和 HMDB51 数据集上进行了广泛测试。通过调整 `alpha`、`beta` 和 `theta` 参数，可以在隐私保护和识别精度之间取得最佳平衡。

推荐配置：
- `alpha=0.2, beta=0.3, theta=0.2`：适合大多数场景
- `alpha=0.2, beta=0.2, theta=0.1-0.4`：可调节隐私保护强度

## 🔧 高级功能

### 学习率调度

支持多种学习率调度策略：

```bash
--lr_scheduler loss_based  # 基于损失的调度
--warmup                   # 启用预热
--warmup_array 0.1 0.2 0.4 0.6 0.8 1.0  # 预热阶段
```

### 数据增强

```bash
--hflip 0 1                # 水平翻转概率
--cropping_facs 0.8 0.9    # 裁剪因子
--weak_aug                 # 使用弱增强
--aspect_ratio_aug         # 启用宽高比增强
```

### 验证策略

```bash
--val_int 10               # 每 10 个 epoch 验证一次
--num_modes 5              # 时序增强模式数
--fix_skip 2               # 固定帧跳跃率
```

## 📝 日志和检查点

训练日志和模型检查点将保存在：

- 日志：`ucf_logs/`, `hmdb_logs/`, `vispr_logs/`
- 检查点：`ckpt/<model_name>/<run_id>/`

## 🤝 贡献

欢迎提交问题和拉取请求！如果您发现任何 bug 或有改进建议，请随时提出。

## 📄 引用

如果您在研究中使用了本代码，请引用我们的论文：

```bibtex
@article{stegavar2025,
  title={StegaVAR: Privacy-Preserving Video Action Recognition via Steganographic Domain Analysis},
  author={Your Name},
  journal={Conference/Journal Name},
  year={2025}
}
```

## 📜 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 👨‍💻 作者

Copyright (c) 2025 Lixin Chen

## 🙏 致谢

本项目基于以下优秀工作：

- UCF101 和 HMDB51 数据集
- PyTorch 深度学习框架
- 相关隐写术和视频分析研究

## 📞 联系方式

如有问题或合作意向，请通过 GitHub Issues 联系我们。

---

**注意**：使用本代码进行研究或应用时，请确保遵守相关的数据隐私和伦理规范。
