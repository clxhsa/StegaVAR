import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2

def extract_frames(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        # 读取视频帧
        ret, frame = cap.read()
        if not ret:
            break
        # 将帧添加到列表中
        frames.append(frame)

    # 释放视频捕获对象
    cap.release()
    return frames

def get_video_fps(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 释放视频捕获对象
    cap.release()

    return fps

def get_video_total_frame(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频的帧率
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # 释放视频捕获对象
    cap.release()

    return count

def save_frame(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 保存每一帧图片
        frame_filename = f'{save_path}/{frame_count}.jpg'
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
        # print(f"{frame_count} saved")

    cap.release()
    print(f'Total frames saved: {frame_count}')


def process_video(video_path):
    if not video_path.endswith(".avi"):
        return f"Skip non-avi file: {video_path}"

    save_path = video_path[:-4]
    save_frame(video_path, save_path)
    return f"Saved frames for {video_path}"

if __name__ == "__main__":
    # 根目录包含类别子目录，每个类别下存放 .avi 视频
    video_root = '/home/chenlixin/StegaVAR/dataset/VAR/hmdb51/hmdb51'
    max_workers = max(1, (os.cpu_count() or 4) - 1)

    video_paths = []
    for class_dir in os.listdir(video_root):
        c_path = os.path.join(video_root, class_dir)
        if not os.path.isdir(c_path):
            continue
        for video in os.listdir(c_path):
            if video.endswith('.avi'):
                video_paths.append(os.path.join(c_path, video))

    print(f"Found {len(video_paths)} videos, using {max_workers} threads")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_video = {executor.submit(process_video, vp): vp for vp in video_paths}
        for idx, future in enumerate(as_completed(future_to_video), 1):
            video = future_to_video[future]
            try:
                msg = future.result()
                print(f"[{idx}/{len(video_paths)}] {msg}")
            except Exception as exc:
                print(f"[{idx}/{len(video_paths)}] {video} generated an exception: {exc}")
