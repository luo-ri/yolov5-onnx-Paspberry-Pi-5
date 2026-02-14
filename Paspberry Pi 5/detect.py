import cv2
import numpy as np
import onnxruntime as ort
import json
import os
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser(description='YOLO 目标检测')
parser.add_argument('--model', type=str, default='best.onnx', help='模型文件路径')
parser.add_argument('--mode', type=str, choices=['image', 'video', 'camera'],
                    default='image', help='检测模式: image(图像), video(视频), camera(摄像头)')
parser.add_argument('--input', type=str, default='', help='输入文件路径（图像或视频）')
parser.add_argument('--output', type=str, default='', help='输出文件路径')
parser.add_argument('--camera', type=int, default=0, help='摄像头设备ID')
parser.add_argument('--skip-frames', type=int, default=15, help='跳帧数（每N帧处理一帧）')
parser.add_argument('--provider', type=str, choices=['cpu', 'gpu', 'tensorrt'],
                    default='cpu', help='ONNX Runtime 推理后端')

args = parser.parse_args()

# 1. 加载模型（优化配置）
print(f"加载模型: {args.model}")
if not os.path.exists(args.model):
    print(f"错误: 模型文件 '{args.model}' 不存在")
    exit(1)

# 根据指定提供商创建会话
providers = []
if args.provider == 'gpu':
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    print("使用 GPU 推理")
elif args.provider == 'tensorrt':
    providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    print("使用 TensorRT 推理")
else:
    providers = ['CPUExecutionProvider']
    print("使用 CPU 推理")

try:
    session = ort.InferenceSession(args.model, providers=providers)
    print("模型加载成功")
except Exception as e:
    print(f"警告: 使用指定的提供商失败 ({e})，回退到 CPU")
    session = ort.InferenceSession(args.model)
    print("模型加载成功（CPU 模式）")

# 2. 从模型获取类别名称
def get_class_names_from_onnx(session):
    """从 ONNX 模型元数据中获取类别名称"""
    try:
        metadata = session.get_modelmeta()
        if metadata and metadata.custom_metadata_map:
            class_names_str = metadata.custom_metadata_map.get('names')
            if class_names_str:
                # 尝试多种格式解析
                try:
                    class_names = json.loads(class_names_str)
                    if isinstance(class_names, dict):
                        return {int(k): v for k, v in class_names.items()}
                    elif isinstance(class_names, list):
                        return {i: name for i, name in enumerate(class_names)}
                except json.JSONDecodeError:
                    # 尝试 Python eval 格式
                    try:
                        class_names = eval(class_names_str)
                        if isinstance(class_names, dict):
                            return {int(k): v for k, v in class_names.items()}
                        elif isinstance(class_names, list):
                            return {i: name for i, name in enumerate(class_names)}
                    except:
                        pass

                # 尝试按行分割
                lines = class_names_str.strip().split('\n')
                if len(lines) > 1:
                    return {i: line.strip() for i, line in enumerate(lines) if line.strip()}

    except Exception as e:
        pass

    return {}

CLASS_NAMES = get_class_names_from_onnx(session)

def get_class_name(class_id):
    """获取类别名称，如果未定义则返回类别ID"""
    return CLASS_NAMES.get(class_id, f'class_{class_id}')

# 3. 获取输入输出名称和尺寸
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 尝试获取模型输入尺寸
try:
    model_input_shape = session.get_inputs()[0].shape
    # 处理动态输入尺寸的情况
    if model_input_shape[2] == 'height' or model_input_shape[3] == 'width':
        # 动态输入尺寸，使用默认值
        model_input_size = 640
        print("模型支持动态输入尺寸，使用默认值: 640x640")
    else:
        # 固定输入尺寸
        model_input_size = int(model_input_shape[2]) if len(model_input_shape) > 2 else 640
        print(f"模型输入尺寸: {model_input_size}x{model_input_size}")
except Exception as e:
    # 如果无法获取尺寸，使用默认值
    model_input_size = 640
    print(f"无法确定模型输入尺寸，使用默认值: 640x640 (错误: {e})")

# 3. 图像预处理函数
def preprocess(img):
    # 使用模型固定的输入尺寸
    img = cv2.resize(img, (model_input_size, model_input_size))
    # 转换颜色通道 (BGR -> RGB) 和维度 (HWC -> CHW)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = img.astype(np.float32) / 255.0  # 归一化
    img = np.expand_dims(img, axis=0)    # 增加 batch 维度
    return img

# 4. 处理输出结果（解析边界框和类别）
def postprocess(outputs, original_img_shape):
    """解析 YOLO 模型的输出结果"""
    predictions = outputs[0]  # 获取预测结果 (1, num_detections, 85) 格式

    # 转置并处理数据
    predictions = predictions[0]  # 移除 batch 维度

    # 设置置信度阈值
    conf_threshold = 0.5

    detections = []
    for detection in predictions:
        # YOLO 格式: [x, y, w, h, confidence, class_scores...]
        x, y, w, h = detection[0:4]
        confidence = detection[4]

        # 获取类别分数
        class_scores = detection[5:]
        class_id = np.argmax(class_scores)
        class_score = class_scores[class_id]

        # 过滤低置信度检测
        if confidence > conf_threshold:
            # 将边界框坐标转换回原始图像尺寸
            original_h, original_w = original_img_shape[:2]
            scale_x = original_w / 640
            scale_y = original_h / 640

            x1 = int((x - w/2) * scale_x)
            y1 = int((y - h/2) * scale_y)
            x2 = int((x + w/2) * scale_x)
            y2 = int((y + h/2) * scale_y)

            detections.append({
                'class_id': int(class_id),
                'confidence': float(confidence),
                'bbox': [x1, y1, x2, y2]
            })

    return detections

def nms(detections, iou_threshold=0.5):
    """非极大值抑制，去除重复的检测框"""
    if len(detections) == 0:
        return []

    # 按置信度降序排序
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

    keep = []
    while len(detections) > 0:
        # 保留置信度最高的检测框
        current = detections[0]
        keep.append(current)

        # 计算剩余检测框与当前检测框的 IoU
        rest = []
        for det in detections[1:]:
            if det['class_id'] != current['class_id']:
                # 不同类别，直接保留
                rest.append(det)
            else:
                iou = calculate_iou(current['bbox'], det['bbox'])
                if iou < iou_threshold:
                    # IoU 小于阈值，保留
                    rest.append(det)

        detections = rest

    return keep

def calculate_iou(box1, box2):
    """计算两个边界框的交并比 (IoU)"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # 计算交集区域
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # 检查是否有交集
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # 计算并集区域
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def draw_detections(img, detections):
    """在图像上绘制检测结果"""
    for detection in detections:
        class_id = detection['class_id']
        class_name = get_class_name(class_id)
        x1, y1, x2, y2 = detection['bbox']

        # 在图像上绘制检测结果
        color = (0, 255, 0)  # 绿色
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {detection['confidence']:.2f}"
        cv2.putText(img, label,
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

def detect_image(img_path, output_path='result.jpg'):
    """对单张图像进行检测"""
    print(f"\n检测图像: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print(f"错误: 无法读取图像文件 '{img_path}'")
        return None

    input_data = preprocess(img)
    outputs = session.run([output_name], {input_name: input_data})

    detections = postprocess(outputs, img.shape)
    detections = nms(detections, iou_threshold=0.5)

    # 打印检测结果
    for i, detection in enumerate(detections, 1):
        class_id = detection['class_id']
        class_name = get_class_name(class_id)
        print(f"\n目标 {i}:")
        print(f"  类别: {class_name} (ID: {class_id})")
        print(f"  置信度: {detection['confidence']:.4f}")
        x1, y1, x2, y2 = detection['bbox']
        print(f"  边界框: [{x1}, {y1}, {x2}, {y2}]")

    # 绘制并保存结果
    img = draw_detections(img, detections)
    cv2.imwrite(output_path, img)
    print(f"\n检测结果已保存到: {output_path}")
    return img

def detect_video(video_path, output_path='output.mp4', skip_frames=0):
    """对视频文件进行检测"""
    print(f"\n检测视频: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 '{video_path}'")
        return

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 设置视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    print("开始处理视频...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"处理中... 已处理 {frame_count} 帧")

        input_data = preprocess(frame)
        outputs = session.run([output_name], {input_name: input_data})

        detections = postprocess(outputs, frame.shape)
        detections = nms(detections, iou_threshold=0.5)

        # 绘制检测结果
        frame = draw_detections(frame, detections)
        out.write(frame)

    cap.release()
    out.release()
    print(f"\n视频处理完成！共处理 {frame_count} 帧")
    print(f"结果已保存到: {output_path}")

def detect_camera(camera_id=0, skip_frames=0):
    """对摄像头进行实时检测"""
    print(f"\n打开摄像头: {camera_id}")
    print("按 'q' 键退出...")

    if skip_frames > 0:
        print(f"跳帧模式: 每 {skip_frames + 1} 帧处理一帧")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"错误: 无法打开摄像头 {camera_id}")
        return

    # 获取摄像头实际帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"摄像头实际帧率: {fps:.2f} FPS")

    frame_count = 0
    processed_count = 0
    total_detections = 0

    # 用于计算实时帧率
    start_time = cv2.getTickCount()
    last_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 跳帧优化
        if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
            # 使用上一帧的检测结果
            frame = draw_detections(frame, last_detections)

            # 计算显示帧率
            elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

            info_text = f"FPS: {avg_fps:.1f} (skipped) | Detections: {total_detections}"
            cv2.putText(frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('YOLO Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # 计算当前帧的处理时间
        loop_start = cv2.getTickCount()

        input_data = preprocess(frame)
        outputs = session.run([output_name], {input_name: input_data})

        detections = postprocess(outputs, frame.shape)
        detections = nms(detections, iou_threshold=0.5)

        last_detections = detections
        total_detections += len(detections)
        processed_count += 1

        # 绘制检测结果
        frame = draw_detections(frame, detections)

        # 计算处理时间和实时帧率
        loop_end = cv2.getTickCount()
        loop_time = (loop_end - loop_start) / cv2.getTickFrequency()
        current_fps = 1.0 / loop_time if loop_time > 0 else 0

        # 计算平均帧率
        elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # 显示帧信息
        mode_str = " | Skip" if skip_frames > 0 else ""
        info_text = f"FPS: {current_fps:.1f} | Avg: {avg_fps:.1f}{mode_str} | Detections: {total_detections}"
        cv2.putText(frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"Processed: {processed_count}/{frame_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('YOLO Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 显示统计信息
    cap.release()
    cv2.destroyAllWindows()

    total_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
    final_avg_fps = frame_count / total_time if total_time > 0 else 0

    print(f"\n摄像头检测已停止")
    print(f"总帧数: {frame_count}")
    print(f"处理帧数: {processed_count}")
    print(f"总时间: {total_time:.2f} 秒")
    print(f"平均帧率: {final_avg_fps:.2f} FPS")
    print(f"总检测数: {total_detections}")

# 根据参数选择检测方式
if args.mode == 'image':
    detect_image(args.input, args.output)
elif args.mode == 'video':
    detect_video(args.input, args.output, args.skip_frames)
elif args.mode == 'camera':
    detect_camera(args.camera, args.skip_frames)
else:
    print("未知模式，请使用 --mode 参数选择检测方式")

