import imageio
import os
from PIL import Image
import numpy as np

input_path = "/data/workspace/Anchor-Frame-Bridging-for-Coherent-First-Last-Frame-Video-Generation-AFB/assets/AFB_demo.mp4"
output_path = "/data/workspace/Anchor-Frame-Bridging-for-Coherent-First-Last-Frame-Video-Generation-AFB/assets/AFB_demo.gif"

# 读取视频
reader = imageio.get_reader(input_path)
fps = reader.get_meta_data().get('fps', 30)

# 更激进的压缩
frames = []
skip = 6  # 每6帧取1帧
max_frames = 50  # 最多50帧
count = 0

for i, frame in enumerate(reader):
    if i % skip == 0:
        # 缩小尺寸到原来的 25%
        h, w = frame.shape[:2]
        new_h, new_w = int(h * 0.25), int(w * 0.25)
        img = Image.fromarray(frame)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        frames.append(np.array(img))
        count += 1
        if count >= max_frames:
            break

# 保存为 GIF
imageio.mimsave(output_path, frames, fps=8, loop=0)
print(f"GIF saved to {output_path}")
print(f"Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
print(f"Frames: {len(frames)}")
