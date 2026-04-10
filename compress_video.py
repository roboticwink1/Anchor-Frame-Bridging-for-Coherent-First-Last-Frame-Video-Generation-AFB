import subprocess
import os
import imageio_ffmpeg

input_path = "/data/workspace/Anchor-Frame-Bridging-for-Coherent-First-Last-Frame-Video-Generation-AFB/assets/AFB_demo.mp4"
output_path = "/data/workspace/Anchor-Frame-Bridging-for-Coherent-First-Last-Frame-Video-Generation-AFB/assets/AFB_demo_small.mp4"

ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

# 压缩视频到 10MB 左右，保持清晰度
cmd = [
    ffmpeg_exe, "-y",
    "-i", input_path,
    "-c:v", "libx264",
    "-crf", "30",  # 压缩质量
    "-preset", "slow",
    "-vf", "scale=640:-2",  # 缩放宽度到640
    "-an",  # 去除音频
    output_path
]

print("Compressing video...")
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print("Error:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
else:
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"Done! Output: {output_path}")
    print(f"Size: {size_mb:.2f} MB")
