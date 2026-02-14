import time
import numpy as np
import onnxruntime as ort


def benchmark(model_path, num_runs=100):
    session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    input_name = session.get_inputs()[0].name

    # 随机生成输入数据
    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)

    # 预热
    for _ in range(10):
        session.run(None, {input_name: dummy_input})

    # 正式测试
    start = time.time()
    for _ in range(num_runs):
        session.run(None, {input_name: dummy_input})
    end = time.time()

    avg_latency = (end - start) / num_runs * 1000  # ms
    print(f"{model_path}: 平均延迟 {avg_latency:.2f} ms ({1000 / avg_latency:.1f} FPS)")
    return avg_latency


# 测试两个模型
benchmark("weights/roda_best_pruned.onnx")
benchmark("weights/road_best.onnx")