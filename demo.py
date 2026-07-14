import cv2
from ultralytics import YOLO
import torch
import numpy as np

# ==========================================
# 0. 数学工具函数：计算角度
# ==========================================
def calculate_angle(a,b,c):
    """
        计算三个点构成的夹角 (以 b 为顶点)

        参数:
        a, b, c: 分别代表三个关键点的坐标，格式为 [x, y] 或 (x, y)
                 例如: a = [100, 200]

        返回:
        angle: 计算出的角度，范围在 0.0 到 180.0 度之间
        """
    # 将python列表转化为Numpy数组
    a = np.array(a) #臀部
    b = np.array(b) #膝盖
    c = np.array(c) #脚踝
    vector_ba = a - b;
    vector_bc = c - b;
    #点乘
    dot_product = np.dot(vector_ba, vector_bc)
    #模长
    norm_ba = np.linalg.norm(vector_ba)
    norm_bc = np.linalg.norm(vector_bc)
    #夹角
    cos_angle = dot_product / (norm_ba*norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    # 6. 使用反余弦函数求出弧度 (Radian)
    angle_rad = np.arccos(cos_angle)

    # 7. 将弧度转换为我们熟悉的角度 (Degree)
    # 1 弧度 ≈ 57.3 度，公式为：弧度 * 180 / π
    angle_deg = np.degrees(angle_rad)

    return angle_deg

# ==========================================
# 1. 环境与模型初始化
# ==========================================
# 检查 Mac M2 的 MPS 加速
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"🚀 使用设备: {device}")

# 直接加载YOLO官方模型
print(f"正在加载模型yolov8n-pose...")
model = YOLO('yolov8n-pose.pt')

# ==========================================
# 2. 视频读取初始化
# ==========================================
video_path = 'my_squat.mp4'
cap = cv2.VideoCapture(video_path)

#状态机
counter = 0
stage = None
# 检查视频是否成功打开
if not cap.isOpened():
    print(f"❌ 错误：无法打开视频文件 {video_path}")
    exit()

print("🎬 开始处理视频... 按 'q' 键退出。")

# ==========================================
# 3. 逐帧处理视频循环
# ==========================================
frame_count = 0
while True:
    success, frame = cap.read()
    if not success:
        print(f"Over")
        break
    frame_count += 1
    # 运行 YOLO 推理
    # verbose=False 可以关闭终端里每一帧的打印信息，让终端更清爽
    results = model(frame , device = device , verbose = False)
    # 提取结果对象 (因为我们只传了一帧图像，所以取 [0])
    result = results[0]

    # ==========================================
    # 4. 提取“左膝盖”"左臀"“左脚踝”坐标的核心逻辑
    # ==========================================
    # 检查是否检测到了人 (如果有 boxes，说明有人)
    if len(result.boxes) > 0:
        # 获取第一个人的关键点数据
        # result.keypoints.xy 包含了坐标数据，形状通常是 (人数, 17, 2)
        # [0] 代表画面中的第一个人
        keypoints = result.keypoints.xy[0]

        # COCO 格式中，左膝盖的索引是 13
        left_hip_index = 11
        left_knee_index = 13
        left_ankle_index = 15
        #列表形式，提取三个关键点的坐标
        hip = [int(keypoints[left_hip_index][0].item()), int(keypoints[left_hip_index][1].item())]
        knee = [int(keypoints[left_knee_index][0].item()) , int(keypoints[left_knee_index][1].item())]
        ankle = [int(keypoints[left_ankle_index][0].item()), int(keypoints[left_ankle_index][1].item())]
        # 计算膝盖夹角
        angle = calculate_angle(hip, knee, ankle)
        #判断并计数
        # 设定阈值：站直大于 150 度，蹲下小于 90 度
        if angle > 150:
            if stage == 'down':
                counter += 1
                print(f"🎉 成功完成 1 个深蹲！当前总数: {counter}")
            stage = 'up'
        if angle < 90:
            stage = 'down'
        # --- 可视化 (画图) ---
        # 1. 画出这三个关键点 (红色实心圆)
        cv2.circle(frame, tuple(hip), 5, (0, 0, 255), -1)
        cv2.circle(frame, tuple(knee), 5, (0, 0, 255), -1)
        cv2.circle(frame, tuple(ankle), 5, (0, 0, 255), -1)

        # 2. 用线把这三个点连起来 (绿色线段，线宽 2)
        cv2.line(frame, tuple(hip), tuple(knee), (0, 255, 0), 2)
        cv2.line(frame, tuple(knee), tuple(ankle), (0, 255, 0), 2)

        # 3. 把计算出的角度写在膝盖旁边
        # 格式化角度，保留一位小数
        angle_text = f"{angle:.1f} deg"
        # cv2.putText 参数：图像, 文字, 坐标(稍微偏移一点避免挡住膝盖), 字体, 缩放, 颜色(蓝色), 线宽
        cv2.putText(frame, angle_text, (knee[0] + 15, knee[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        # ==========================================
        # 🌟 新增：在画面左上角显示计数面板
        # ==========================================
        # 画一个半透明的黑色背景框，让白色文字更清晰
        cv2.rectangle(frame, (10, 10), (250, 120), (0, 0, 0), -1)

        # 显示当前状态 (UP / DOWN)
        cv2.putText(frame, f"Stage: {stage}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # 显示深蹲次数
        cv2.putText(frame, f"Reps: {counter}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # ==========================================
    # 5. 画面展示与退出逻辑
    # ==========================================
    # 显示带有标记的画面
    cv2.imshow('Squat_Tracker', frame)
    # 等待 1 毫秒，如果用户按下 'q' 键，则跳出循环
    # 注意：处理视频时不能用 waitKey(0)，否则画面会卡在第一帧
    if cv2.waitKey(30) & 0xFF == ord('q'):
        print("⏹️ 用户手动中断。")
        break

# ==========================================
# 6. 资源释放
# ==========================================
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)  # Mac 专属防卡死小妙招















