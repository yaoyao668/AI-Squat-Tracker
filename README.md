# 🏋️‍♂️ AI Squat Tracker (AI 深蹲计数器)

基于 **YOLOv8-Pose** 关键点检测的轻量级深蹲计数与动作分析工具。通过计算机视觉技术，实时计算人体膝关节角度，并利用状态机逻辑实现精准的深蹲计数。

![Demo Screenshot](./assets/demo_screenshot.png)

## ✨ 核心功能 (Features)

- **实时姿态估计**：使用 Ultralytics YOLOv8n-pose 模型，极速提取人体 17 个关键点。
- **精准角度计算**：利用 Numpy 向量点积公式，实时计算臀部-膝盖-脚踝构成的夹角。
- **智能状态机计数**：
  - 站立阈值 (Up Threshold): `> 150°`
  - 下蹲阈值 (Down Threshold): `< 90°`
  - 结合状态机 (State Machine) 逻辑，完美过滤无效抖动，实现精准计数。
- **可视化 UI 面板**：实时在视频画面中渲染骨架连线、动态角度以及计数统计面板。
- **硬件加速支持**：自动检测并支持 Apple Silicon (Mac M1/M2 MPS) 及 NVIDIA GPU (CUDA) 加速。

## 🛠️ 安装指南 (Installation)

1. 克隆本仓库：
```bash
git clone https://github.com/your-username/AI-Squat-Tracker.git
cd AI-Squat-Tracker
```

2. 安装依赖包：
```bash
pip install -r requirements.txt
```

## 🚀 快速开始 (Usage)

将你需要测试的深蹲视频放入项目目录，并修改 `demo.py` 中的 `video_path`，然后运行：

```bash
python demo.py
```

*操作提示：在视频播放窗口中，按下键盘上的 `q` 键即可随时退出程序。*

## 🧠 算法原理 (How it works)

1. **关键点提取**：从 YOLOv8 的输出张量中提取索引为 `11` (Hip), `13` (Knee), `15` (Ankle) 的三个坐标点。
2. **向量夹角**：
   $$ \cos(\theta) = \frac{\vec{BA} \cdot \vec{BC}}{|\vec{BA}| \times |\vec{BC}|} $$
3. **状态切换**：当角度小于 90° 时进入 `down` 状态；当角度恢复到 150° 以上且前一状态为 `down` 时，判定为完成一次深蹲，计数器 `+1` 并重置状态为 `up`。

## 🤝 贡献与反馈 (Contributing)
欢迎提交 Pull Request 或 Issue 探讨交流！如果这个项目对你有帮助，请给我一个 ⭐️ Star！