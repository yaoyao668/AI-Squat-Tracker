import cv2

def is_video_supported(video_path):
    cap = cv2.VideoCapture(video_path)
    # 1. 检查是否能打开
    if not cap.isOpened():
        cap.release()
        return False, "无法打开：格式/编解码器不支持或文件损坏"
    # 2. 尝试读一帧（验证能否解码）
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return False, "能打开但无法解码：编码不支持"
    return True, "格式/编码均支持"

# 测试
supported, msg = is_video_supported("my_squat.mp4")
print(supported, msg)