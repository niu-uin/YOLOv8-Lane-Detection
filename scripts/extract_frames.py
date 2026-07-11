"""
从视频中提取帧用于标注（可跳过已存在的帧）。

用法:
  python scripts/extract_frames.py --video data/videos/harder_challenge_video.mp4 --out data/train_from_video/manual_images --step 30
"""
import argparse
import cv2
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='从视频提取帧')
    parser.add_argument('--video', default='data/videos/harder_challenge_video.mp4')
    parser.add_argument('--out', default='data/train_from_video/manual_images')
    parser.add_argument('--step', type=int, default=10, help='每隔多少帧提取一帧')
    parser.add_argument('--start', type=int, default=0, help='起始帧')
    parser.add_argument('--end', type=int, default=None, help='结束帧')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    existing = set(f.name for f in out_dir.glob('*.jpg'))
    frame_idx = 0
    extracted = 0
    skipped = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx < args.start:
            frame_idx += 1
            continue
        if args.end is not None and frame_idx > args.end:
            break

        if frame_idx % args.step == 0:
            name = f"frame_{frame_idx:06d}.jpg"
            if name in existing:
                skipped += 1
            else:
                cv2.imwrite(str(out_dir / name), frame)
                extracted += 1

        frame_idx += 1

    cap.release()
    print(f"提取完成: 新增 {extracted} 张, 跳过 {skipped} 张 (已存在)")


if __name__ == '__main__':
    main()
