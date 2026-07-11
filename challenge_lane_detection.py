# """
# YOLOv8n-seg 车道线检测 — 完整流水线

# 整体流程:
#   ① YOLOv8n-seg 推理 → 车道线分割掩码 → 二值图
#   ② 形态学去噪 + 过滤小连通域
#   ③ 鸟瞰图透视变换
#   ④ 提取左右车道线中心点（按图像中线分离候选轮廓）
#   ⑤ 二次多项式拟合 + 底部加权（近处点更可靠）
#   ⑥ 帧间自适应平滑（变化大时快速跟随，平稳时强平滑）
#   ⑦ 车道区域填充（绿色多边形，内缩 inet 避免溢出）
#   ⑧ 逆透视变换回原图 → 叠加显示
#   ⑨ 计算曲率半径 & 车辆偏移量
#   ⑩ 开头 _MIN_VALID_STREAK 帧不渲染，避免冷启动闪烁
# """
# import cv2
# import numpy as np
# import glob
# import os
# from ultralytics import YOLO

# # ========================= 相机标定 & 透视变换参数 =========================
# nx = 9                            # 棋盘格内角点 (横向)
# ny = 6                            # 棋盘格内角点 (纵向)
# offset_x = 330                    # 透视变换 dst 四角的 x 偏移
# offset_y = 0                      # 透视变换 dst 四角的 y 偏移

# # ========================= 车道线拟合参数 =========================
# xm_per_pix = 3.7 / 700            # 米/像素 (横向：车道宽 3.7m)
# ym_per_pix = 30 / 720             # 米/像素 (纵向：30m)
# lane_center_default = 640         # 图像水平中心 (1280/2)

# SRC_POINTS = [[580, 460], [700, 460], [210, 720], [1110, 720]]

# last_left_fit  = None             # 上一帧左车道线拟合系数
# last_right_fit = None             # 上一帧右车道线拟合系数

# # ========================= 冷启动 & 兜底 =========================
# # 开头几帧完全检测不到时用这两条竖线，避免 None 崩溃
# _FALLBACK_LEFT_FIT  = np.array([0.0, 0.0, 380.0])
# _FALLBACK_RIGHT_FIT = np.array([0.0, 0.0, 900.0])

# _valid_streak    = 0              # 当前连续有效帧数
# _MIN_VALID_STREAK = 1            # 至少连续 4 帧有效才开始渲染绿色
# _OVERLAY_ALPHA   = 0.0            # 叠加透明度 (渐入)

# # ========================= 模型加载 =========================
# _model_path = 'runs/segment/runs/yolov8m_lane/weights/best.pt'
# if not os.path.exists(_model_path):
#     _model_path = 'models/yolov8n-seg.pt'
# _model = YOLO(_model_path)


# # ========================= ① YOLO 推理 → 二值图 =========================
# def pipeline(img):
#     """YOLO 推理 → 二值分割图 (下半部分)"""
#     h, w = img.shape[:2]

#     results = _model(img, verbose=False, conf=0.45)[0]
#     binary = np.zeros((h, w), dtype=np.uint8)
#     if results.masks is not None:
#         masks = results.masks.data.cpu().numpy()
#         for mask in masks:
#             m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
#             binary = cv2.bitwise_or(binary, (m > 0.5).astype(np.uint8))

#     # 只保留下半部分（上半部分不可能有车道线）
#     binary[:h // 2, :] = 0

#     # 形态学去噪 + 填补孔洞
#     kernel = np.ones((5, 5), np.uint8)
#     binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
#     binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

#     # 高斯模糊软化边缘后重新二值化
#     binary = cv2.GaussianBlur(binary.astype(np.float32), (5, 5), 0)
#     binary = (binary > 0.5).astype(np.uint8)

#     # 过滤面积过小的噪声连通域
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
#     for i in range(1, num_labels):
#         if stats[i, cv2.CC_STAT_AREA] < 300:
#             binary[labels == i] = 0

#     return binary


# # ========================= 标定 & 透视变换 =========================
# def cal_calibrate_params(file_paths):
#     """棋盘格相机标定 → 内参 + 畸变系数"""
#     obj_points, img_points = [], []
#     objp = np.zeros((nx * ny, 3), np.float32)
#     objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
#     for fname in file_paths:
#         img = cv2.imread(fname)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
#         if ret:
#             obj_points.append(objp)
#             img_points.append(corners)
#     ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
#         obj_points, img_points, gray.shape[::-1], None, None)
#     return ret, mtx, dist, rvecs, tvecs


# def img_undistort(img, mtx, dist):
#     """图像去畸变"""
#     return cv2.undistort(img, mtx, dist, None, mtx)


# def cal_perspective_params(img, src_points):
#     """计算透视变换矩阵 M 及其逆矩阵 M_inv"""
#     img_size = (img.shape[1], img.shape[0])
#     src = np.float32(src_points)
#     dst = np.float32([
#         [offset_x,               offset_y],
#         [img_size[0] - offset_x, offset_y],
#         [offset_x,               img_size[1] - offset_y],
#         [img_size[0] - offset_x, img_size[1] - offset_y],
#     ])
#     M     = cv2.getPerspectiveTransform(src, dst)
#     M_inv = cv2.getPerspectiveTransform(dst, src)
#     return M, M_inv


# def img_perspect_transform(img, M):
#     """对图像做透视变换"""
#     img_size = (img.shape[1], img.shape[0])
#     return cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)


# # ========================= ③ 车道线中心点提取 =========================
# def extract_lane_centerlines(binary):
#     """
#     按图像水平中线将候选轮廓分入左右两组 → 各取面积最大者 → 提取每行中心点。

#     原版问题: 直接取面积最大的两个轮廓，不能保证一左一右。
#     本版方案: 只考虑面积前 10 大的轮廓，按 mean_x < mid_x 分到左右组，
#               各组取面积最大轮廓作为该侧车道线。
#     """
#     contours, _ = cv2.findContours(
#         binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)

#     if len(contours) < 2:
#         return None, None

#     h, w = binary.shape
#     mid_x = w // 2

#     left_candidates  = []
#     right_candidates = []
#     for cnt in contours[:10]:
#         if cv2.contourArea(cnt) < 100:
#             break
#         pts    = cnt.reshape(-1, 2)
#         mean_x = np.mean(pts[:, 0])
#         if mean_x < mid_x:
#             left_candidates.append(cnt)
#         else:
#             right_candidates.append(cnt)

#     if not left_candidates or not right_candidates:
#         return None, None

#     lane_left  = left_candidates[0]
#     lane_right = right_candidates[0]

#     # 提取每条车道线的每行中心点 → 供后续多项式拟合
#     lanes = []
#     for lane in [lane_left, lane_right]:
#         pts = lane.reshape(-1, 2)
#         ys  = np.unique(pts[:, 1])
#         center_points = []
#         for y in ys:
#             row      = pts[pts[:, 1] == y]
#             center_x = (np.min(row[:, 0]) + np.max(row[:, 0])) / 2
#             center_points.append([center_x, y])
#         lanes.append(np.array(center_points))

#     return lanes[0], lanes[1]


# # ========================= ④ 多项式拟合 (底部加权) =========================
# def fit_lane_curve(centerline):
#     """
#     二次多项式拟合 x = a*y² + b*y + c。

#     底部（靠近摄像头）的点权重更高，因为近处检测更稳定可靠。
#     少于 30 个点则放弃拟合。
#     """
#     if centerline is None:
#         return None
#     x = centerline[:, 0]
#     y = centerline[:, 1]
#     if len(x) < 30:
#         return None

#     y_min, y_max = np.min(y), np.max(y)
#     weights = (y - y_min) / (y_max - y_min + 1e-6) + 0.2
#     return np.polyfit(y, x, 2, w=weights)


# # ========================= ⑤ 帧间自适应平滑 =========================
# def smooth_fit(current_fit, previous_fit, alpha_stable=0.75, alpha_fast=0.25):
#     """
#     自适应指数平滑。

#     变化大（车道真正移动） → alpha_fast (0.45) 快速跟随，约 3 帧追上
#     变化小（噪声抖动）    → alpha_stable (0.75) 强平滑，约 10 帧追上

#     判断依据：横向位移差 delta = |curr_c - prev_c| (常数项)
#     """
#     if previous_fit is None:
#         return current_fit
#     if current_fit is None:
#         return previous_fit

#     delta = np.abs(current_fit[2] - previous_fit[2])
#     alpha = alpha_fast if delta > 40 else alpha_stable
#     return alpha * previous_fit + (1 - alpha) * current_fit


# # ========================= ⑥ 绿色车道区域填充 =========================
# def fill_lane_poly(img, left_fit, right_fit, inset=12):
#     """
#     在鸟瞰图上绘制绿色车道区域多边形。

#     inset: 左右各向内收缩 inset 像素，避免绿色覆盖到车道线外面。
#     start_y: 从图像 54% 高度开始填充，顶部远处不稳定区域留白。
#     """
#     out_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
#     start_y = int(img.shape[0] * 0.54)
#     end_y   = img.shape[0]

#     left_points  = []
#     right_points = []
#     for y in range(start_y, end_y):
#         lx = left_fit[0] * y * y + left_fit[1] * y + left_fit[2] + inset
#         rx = right_fit[0] * y * y + right_fit[1] * y + right_fit[2] - inset
#         lx = np.clip(lx, 0, img.shape[1] - 1)
#         rx = np.clip(rx, 0, img.shape[1] - 1)
#         if lx < rx:
#             left_points.append([lx, y])
#             right_points.append([rx, y])

#     if left_points:
#         points = np.vstack((left_points, right_points[::-1]))
#         cv2.fillPoly(out_img, [np.int32(points)], (0, 255, 0))

#     return out_img


# # ========================= ⑨ 曲率半径 & 偏移量 =========================
# def cal_radius(img, left_fit, right_fit):
#     """计算并显示车道曲率半径 (米)"""
#     y_vals       = np.linspace(0, img.shape[0] - 1, img.shape[0])
#     left_x_real  = (left_fit[0]  * y_vals ** 2 + left_fit[1]  * y_vals + left_fit[2])  * xm_per_pix
#     right_x_real = (right_fit[0] * y_vals ** 2 + right_fit[1] * y_vals + right_fit[2]) * xm_per_pix
#     y_real       = y_vals * ym_per_pix
#     left_fit_real  = np.polyfit(y_real, left_x_real,  2)
#     right_fit_real = np.polyfit(y_real, right_x_real, 2)
#     y_eval_real    = np.max(y_real)

#     left_R  = ((1 + (2 * left_fit_real[0]  * y_eval_real + left_fit_real[1])  ** 2) ** 1.5) / np.abs(2 * left_fit_real[0])
#     right_R = ((1 + (2 * right_fit_real[0] * y_eval_real + right_fit_real[1]) ** 2) ** 1.5) / np.abs(2 * right_fit_real[0])
#     avg_radius = (left_R + right_R) / 2.0

#     cv2.putText(img, f'Radius of Curvature: {avg_radius:.1f} m',
#                 (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#     return img


# def cal_center_departure(img, left_fit, right_fit, lane_center=lane_center_default):
#     """计算并显示车辆偏离车道中心的距离 (米)"""
#     y_max            = img.shape[0] - 1
#     left_x           = left_fit[0]  * y_max ** 2 + left_fit[1]  * y_max + left_fit[2]
#     right_x          = right_fit[0] * y_max ** 2 + right_fit[1] * y_max + right_fit[2]
#     lane_center_pixel = (left_x + right_x) / 2.0
#     offset_m         = (lane_center_pixel - lane_center) * xm_per_pix

#     if offset_m > 0:
#         text = f'Vehicle is {offset_m:.2f} m right of center'
#     elif offset_m < 0:
#         text = f'Vehicle is {-offset_m:.2f} m left of center'
#     else:
#         text = 'Vehicle is in the center'
#     cv2.putText(img, text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#     return img


# # ========================= 单帧处理 =========================
# def process_single_image(img, mtx, dist, M, M_inv, src_points):
#     """处理一帧图像：①~⑩ 全流程"""
#     global last_left_fit, last_right_fit, _valid_streak, _OVERLAY_ALPHA

#     undist     = img_undistort(img, mtx, dist)
#     binary     = pipeline(undist)
#     warp       = img_perspect_transform(binary, M)
#     left_lane, right_lane = extract_lane_centerlines(warp)
#     raw_left  = fit_lane_curve(left_lane)
#     raw_right = fit_lane_curve(right_lane)

#     # ── 连续有效帧计数 ────────────────────────────────────────────────────────
#     # 只有左右两侧都拟合成功才算有效（both_valid=True）。
#     # 开头 _MIN_VALID_STREAK 帧不渲染绿色，避免冷启动闪烁。
#     both_valid = (raw_left is not None) and (raw_right is not None)
#     if both_valid:
#         _valid_streak += 1
#     else:
#         _valid_streak = 0
#     # ─────────────────────────────────────────────────────────────────────────

#     # None 保护（传入 smooth_fit 时不会崩，但不参与渲染决策）
#     left_fit  = raw_left  if raw_left  is not None else (last_left_fit  if last_left_fit  is not None else _FALLBACK_LEFT_FIT.copy())
#     right_fit = raw_right if raw_right is not None else (last_right_fit if last_right_fit is not None else _FALLBACK_RIGHT_FIT.copy())

#     # 帧间自适应平滑
#     left_fit  = smooth_fit(left_fit,  last_left_fit)
#     right_fit = smooth_fit(right_fit, last_right_fit)
#     last_left_fit  = left_fit
#     last_right_fit = right_fit

#     # ── 冷启动保护：未达到最小连续有效帧 → 返回原图 ──────────────────────────
#     if _valid_streak < _MIN_VALID_STREAK:
#         _OVERLAY_ALPHA = 0.0
#         return undist

#     # 叠透明度线性渐入 (0 → 0.5)
#     _OVERLAY_ALPHA = min(0.5, _OVERLAY_ALPHA + 0.05)
#     # ─────────────────────────────────────────────────────────────────────────

#     warp_color = fill_lane_poly(warp, left_fit, right_fit)
#     inv_warp   = img_perspect_transform(warp_color, M_inv)
#     result     = cal_radius(inv_warp, left_fit, right_fit)
#     result     = cal_center_departure(result, left_fit, right_fit, lane_center=lane_center_default)
#     final      = cv2.addWeighted(undist, 1, result, _OVERLAY_ALPHA, 0)

#     return final


# # ========================= 视频处理 =========================
# def process_video(input_path, output_path, mtx, dist, M, M_inv, src_points):
#     """逐帧处理视频并输出"""
#     cap    = cv2.VideoCapture(input_path)
#     fps    = int(cap.get(cv2.CAP_PROP_FPS))
#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         img_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         result     = process_single_image(img_rgb, mtx, dist, M, M_inv, src_points)
#         result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
#         out.write(result_bgr)

#     cap.release()
#     out.release()
#     print(f"视频处理完成，保存至 {output_path}")


# # ========================= 主程序 =========================
# if __name__ == "__main__":
#     cal_images = glob.glob("data/camera_cal/calibration*.jpg")
#     if cal_images:
#         ret, mtx, dist, rvecs, tvecs = cal_calibrate_params(cal_images)
#         print("相机标定完成，重投影误差：", ret)
#     else:
#         mtx  = np.array([[1.0e3, 0, 640], [0, 1.0e3, 360], [0, 0, 1]], dtype=np.float32)
#         dist = np.zeros(5)

#     test_img  = cv2.imread("data/test_images/straight_lines2.jpg")
#     M, M_inv  = cal_perspective_params(test_img, SRC_POINTS)
#     print("透视变换矩阵计算完成")

#     process_video("data/videos/challenge_video.mp4", "output_challenge_video.mp4",
#                   mtx, dist, M, M_inv, SRC_POINTS)

"""
YOLOv8n-seg 车道线检测 — 完整流水线

整体流程:
  ① YOLOv8n-seg 推理 → 车道线分割掩码 → 二值图
  ② 形态学去噪 + 过滤小连通域
  ③ 鸟瞰图透视变换
  ④ 提取左右车道线中心点（按图像中线分离候选轮廓）
  ⑤ 二次多项式拟合 + 底部加权（近处点更可靠）
  ⑥ 帧间自适应平滑（变化大时快速跟随，平稳时强平滑）
  ⑦ 车道区域填充（绿色多边形，内缩 inet 避免溢出）
  ⑧ 逆透视变换回原图 → 叠加显示
  ⑨ 计算曲率半径 & 车辆偏移量
  ⑩ 开头 _MIN_VALID_STREAK 帧不渲染，避免冷启动闪烁
"""
import cv2
import numpy as np
import glob
import os
from ultralytics import YOLO

# ========================= 相机标定 & 透视变换参数 =========================
nx = 9                            # 棋盘格内角点 (横向)
ny = 6                            # 棋盘格内角点 (纵向)
offset_x = 330                    # 透视变换 dst 四角的 x 偏移
offset_y = 0                      # 透视变换 dst 四角的 y 偏移

# ========================= 车道线拟合参数 =========================
xm_per_pix = 3.7 / 700            # 米/像素 (横向：车道宽 3.7m)
ym_per_pix = 30 / 720             # 米/像素 (纵向：30m)
lane_center_default = 640         # 图像水平中心 (1280/2)

SRC_POINTS = [[580, 460], [700, 460], [210, 720], [1110, 720]]

last_left_fit  = None             # 上一帧左车道线拟合系数
last_right_fit = None             # 上一帧右车道线拟合系数

# ========================= 冷启动 & 兜底 =========================
# 开头几帧完全检测不到时用这两条竖线，避免 None 崩溃
_FALLBACK_LEFT_FIT  = np.array([0.0, 0.0, 380.0])
_FALLBACK_RIGHT_FIT = np.array([0.0, 0.0, 900.0])

_valid_streak    = 0              # 当前连续有效帧数
_MIN_VALID_STREAK = 1            # 至少连续 4 帧有效才开始渲染绿色
_OVERLAY_ALPHA   = 0.0            # 叠加透明度 (渐入)

# ========================= 模型加载 =========================
_model_path = 'runs/segment/runs/yolov8m_lane/weights/best.pt'
if not os.path.exists(_model_path):
    _model_path = 'models/yolov8n-seg.pt'
_model = YOLO(_model_path)


# ========================= ① YOLO 推理 → 二值图 =========================
def pipeline(img):
    """YOLO 推理 → 二值分割图 (下半部分)"""
    h, w = img.shape[:2]

    results = _model(img, verbose=False, conf=0.45)[0]
    binary = np.zeros((h, w), dtype=np.uint8)
    if results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        for mask in masks:
            m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            binary = cv2.bitwise_or(binary, (m > 0.5).astype(np.uint8))

    # 只保留下半部分（上半部分不可能有车道线）
    binary[:h // 2, :] = 0

    # 形态学去噪 + 填补孔洞
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 高斯模糊软化边缘后重新二值化
    binary = cv2.GaussianBlur(binary.astype(np.float32), (5, 5), 0)
    binary = (binary > 0.5).astype(np.uint8)

    # 过滤面积过小的噪声连通域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 300:
            binary[labels == i] = 0

    return binary


# ========================= 标定 & 透视变换 =========================
def cal_calibrate_params(file_paths):
    """棋盘格相机标定 → 内参 + 畸变系数"""
    obj_points, img_points = [], []
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    for fname in file_paths:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret:
            obj_points.append(objp)
            img_points.append(corners)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs


def img_undistort(img, mtx, dist):
    """图像去畸变"""
    return cv2.undistort(img, mtx, dist, None, mtx)


def cal_perspective_params(img, src_points):
    """计算透视变换矩阵 M 及其逆矩阵 M_inv"""
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(src_points)
    dst = np.float32([
        [offset_x,               offset_y],
        [img_size[0] - offset_x, offset_y],
        [offset_x,               img_size[1] - offset_y],
        [img_size[0] - offset_x, img_size[1] - offset_y],
    ])
    M     = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    return M, M_inv


def img_perspect_transform(img, M):
    """对图像做透视变换"""
    img_size = (img.shape[1], img.shape[0])
    return cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)


# ========================= ③ 车道线中心点提取 =========================
def extract_lane_centerlines(binary):
    """
    按图像水平中线将候选轮廓分入左右两组 → 各取面积最大者 → 提取每行中心点。

    原版问题: 直接取面积最大的两个轮廓，不能保证一左一右。
    本版方案: 只考虑面积前 10 大的轮廓，按 mean_x < mid_x 分到左右组，
              各组取面积最大轮廓作为该侧车道线。
    """
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if len(contours) < 2:
        return None, None

    h, w = binary.shape
    mid_x = w // 2

    left_candidates  = []
    right_candidates = []
    for cnt in contours[:10]:
        if cv2.contourArea(cnt) < 100:
            break
        pts    = cnt.reshape(-1, 2)
        mean_x = np.mean(pts[:, 0])
        if mean_x < mid_x:
            left_candidates.append(cnt)
        else:
            right_candidates.append(cnt)

    if not left_candidates or not right_candidates:
        return None, None

    lane_left  = left_candidates[0]
    lane_right = right_candidates[0]

    # 提取每条车道线的每行中心点 → 供后续多项式拟合
    lanes = []
    for lane in [lane_left, lane_right]:
        pts = lane.reshape(-1, 2)
        ys  = np.unique(pts[:, 1])
        center_points = []
        for y in ys:
            row      = pts[pts[:, 1] == y]
            center_x = (np.min(row[:, 0]) + np.max(row[:, 0])) / 2
            center_points.append([center_x, y])
        lanes.append(np.array(center_points))

    return lanes[0], lanes[1]


# ========================= ④ 多项式拟合 (底部加权) =========================
def fit_lane_curve(centerline):
    """
    二次多项式拟合 x = a*y² + b*y + c。

    底部（靠近摄像头）的点权重更高，因为近处检测更稳定可靠。
    少于 30 个点则放弃拟合。
    """
    if centerline is None:
        return None
    x = centerline[:, 0]
    y = centerline[:, 1]
    if len(x) < 30:
        return None

    y_min, y_max = np.min(y), np.max(y)
    weights = (y - y_min) / (y_max - y_min + 1e-6) + 0.2
    return np.polyfit(y, x, 2, w=weights)


# ========================= ⑤ 拟合结果合理性校验 =========================
def is_fit_valid(left_fit, right_fit, img_h=720, img_w=1280,
                 min_lane_width_px=200, max_lane_width_px=900):
    """
    校验左右拟合曲线是否合理：
      1. 在图像底部，左线 x < 右线 x（不交叉）
      2. 底部车道宽度在合理范围内 [min_lane_width_px, max_lane_width_px]
      3. 左右线的二次项系数差异不能过大（防止一条线畸变）
    """
    if left_fit is None or right_fit is None:
        return False
    y_bot = img_h - 1
    y_mid = img_h // 2
    lx_bot = left_fit[0] * y_bot**2 + left_fit[1] * y_bot + left_fit[2]
    rx_bot = right_fit[0] * y_bot**2 + right_fit[1] * y_bot + right_fit[2]
    lx_mid = left_fit[0] * y_mid**2 + left_fit[1] * y_mid + left_fit[2]
    rx_mid = right_fit[0] * y_mid**2 + right_fit[1] * y_mid + right_fit[2]
    width_bot = rx_bot - lx_bot
    width_mid = rx_mid - lx_mid
    if width_bot < min_lane_width_px or width_bot > max_lane_width_px:
        return False
    if width_mid < min_lane_width_px or width_mid > max_lane_width_px:
        return False
    if lx_bot >= rx_bot or lx_mid >= rx_mid:  # 交叉
        return False
    # 二次项系数不应相差过大（防止某条线极度弯曲）
    if abs(left_fit[0] - right_fit[0]) > 5e-3:
        return False
    return True


# ========================= ⑤ 帧间自适应平滑 =========================
def smooth_fit(current_fit, previous_fit, alpha_stable=0.82, alpha_fast=0.2):
    """
    自适应指数平滑。

    变化大（车道真正移动） → alpha_fast (0.45) 快速跟随，约 3 帧追上
    变化小（噪声抖动）    → alpha_stable (0.82) 强平滑，约 10 帧追上

    判断依据：横向位移差 delta = |curr_c - prev_c| (常数项)
    """
    if previous_fit is None:
        return current_fit
    if current_fit is None:
        return previous_fit

    delta = np.abs(current_fit[2] - previous_fit[2])
    alpha = alpha_fast if delta > 40 else alpha_stable
    return alpha * previous_fit + (1 - alpha) * current_fit


# ========================= ⑥ 绿色车道区域填充 =========================
def fill_lane_poly(img, left_fit, right_fit, inset=12):
    """
    在鸟瞰图上绘制绿色车道区域多边形。

    inset: 左右各向内收缩 inset 像素，避免绿色覆盖到车道线外面。
    start_y: 从图像 54% 高度开始填充，顶部远处不稳定区域留白。
    """
    out_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    start_y = int(img.shape[0] * 0.54)
    end_y   = img.shape[0]

    left_points  = []
    right_points = []
    for y in range(start_y, end_y):
        lx = left_fit[0] * y * y + left_fit[1] * y + left_fit[2] + inset
        rx = right_fit[0] * y * y + right_fit[1] * y + right_fit[2] - inset
        lx = np.clip(lx, 0, img.shape[1] - 1)
        rx = np.clip(rx, 0, img.shape[1] - 1)
        if lx < rx:
            left_points.append([lx, y])
            right_points.append([rx, y])

    if left_points:
        points = np.vstack((left_points, right_points[::-1]))
        cv2.fillPoly(out_img, [np.int32(points)], (0, 255, 0))

    return out_img


# ========================= ⑨ 曲率半径 & 偏移量 =========================
def cal_radius(img, left_fit, right_fit):
    """计算并显示车道曲率半径 (米)"""
    y_vals       = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_x_real  = (left_fit[0]  * y_vals ** 2 + left_fit[1]  * y_vals + left_fit[2])  * xm_per_pix
    right_x_real = (right_fit[0] * y_vals ** 2 + right_fit[1] * y_vals + right_fit[2]) * xm_per_pix
    y_real       = y_vals * ym_per_pix
    left_fit_real  = np.polyfit(y_real, left_x_real,  2)
    right_fit_real = np.polyfit(y_real, right_x_real, 2)
    y_eval_real    = np.max(y_real)

    left_R  = ((1 + (2 * left_fit_real[0]  * y_eval_real + left_fit_real[1])  ** 2) ** 1.5) / np.abs(2 * left_fit_real[0])
    right_R = ((1 + (2 * right_fit_real[0] * y_eval_real + right_fit_real[1]) ** 2) ** 1.5) / np.abs(2 * right_fit_real[0])
    avg_radius = (left_R + right_R) / 2.0

    cv2.putText(img, f'Radius of Curvature: {avg_radius:.1f} m',
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img


def cal_center_departure(img, left_fit, right_fit, lane_center=lane_center_default):
    """计算并显示车辆偏离车道中心的距离 (米)"""
    y_max            = img.shape[0] - 1
    left_x           = left_fit[0]  * y_max ** 2 + left_fit[1]  * y_max + left_fit[2]
    right_x          = right_fit[0] * y_max ** 2 + right_fit[1] * y_max + right_fit[2]
    lane_center_pixel = (left_x + right_x) / 2.0
    offset_m         = (lane_center_pixel - lane_center) * xm_per_pix

    if offset_m > 0:
        text = f'Vehicle is {offset_m:.2f} m right of center'
    elif offset_m < 0:
        text = f'Vehicle is {-offset_m:.2f} m left of center'
    else:
        text = 'Vehicle is in the center'
    cv2.putText(img, text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img


# ========================= 单帧处理 =========================
def process_single_image(img, mtx, dist, M, M_inv, src_points):
    """处理一帧图像：①~⑩ 全流程"""
    global last_left_fit, last_right_fit, _valid_streak, _OVERLAY_ALPHA

    undist     = img_undistort(img, mtx, dist)
    binary     = pipeline(undist)
    warp       = img_perspect_transform(binary, M)
    left_lane, right_lane = extract_lane_centerlines(warp)
    raw_left  = fit_lane_curve(left_lane)
    raw_right = fit_lane_curve(right_lane)

    # ── 合理性校验（防止桥墩/阴影导致的异常拟合污染帧间平滑）────────────────
    # 先用合理性校验过滤 raw_left / raw_right
    h_warp, w_warp = warp.shape[:2]
    if not is_fit_valid(raw_left, raw_right, img_h=h_warp, img_w=w_warp):
        # 本帧拟合不可信，强制置 None，沿用上帧结果
        raw_left  = None
        raw_right = None

    # ── 连续有效帧计数 ────────────────────────────────────────────────────────
    # 只有左右两侧都拟合成功才算有效（both_valid=True）。
    # 开头 _MIN_VALID_STREAK 帧不渲染绿色，避免冷启动闪烁。
    both_valid = (raw_left is not None) and (raw_right is not None)
    if both_valid:
        _valid_streak += 1
    else:
        _valid_streak = max(0, _valid_streak - 1)   # 缓慢衰减而非立即归零
    # ─────────────────────────────────────────────────────────────────────────

    # None 保护（传入 smooth_fit 时不会崩，但不参与渲染决策）
    left_fit  = raw_left  if raw_left  is not None else (last_left_fit  if last_left_fit  is not None else _FALLBACK_LEFT_FIT.copy())
    right_fit = raw_right if raw_right is not None else (last_right_fit if last_right_fit is not None else _FALLBACK_RIGHT_FIT.copy())

    # 帧间自适应平滑（异常帧直接用上帧，不参与混合）
    if both_valid:
        left_fit  = smooth_fit(left_fit,  last_left_fit)
        right_fit = smooth_fit(right_fit, last_right_fit)
    # else: 直接使用上帧的 last_fit（已在上方赋给 left_fit / right_fit）
    last_left_fit  = left_fit
    last_right_fit = right_fit

    # ── 冷启动保护：未达到最小连续有效帧 → 返回原图 ──────────────────────────
    if _valid_streak < _MIN_VALID_STREAK:
        if _valid_streak == 0:
            _OVERLAY_ALPHA = 0.0
        return undist

    # 叠透明度线性渐入 (0 → 0.5)
    _OVERLAY_ALPHA = min(0.5, _OVERLAY_ALPHA + 0.05)
    # ─────────────────────────────────────────────────────────────────────────

    warp_color = fill_lane_poly(warp, left_fit, right_fit)
    inv_warp   = img_perspect_transform(warp_color, M_inv)
    result     = cal_radius(inv_warp, left_fit, right_fit)
    result     = cal_center_departure(result, left_fit, right_fit, lane_center=lane_center_default)
    final      = cv2.addWeighted(undist, 1, result, _OVERLAY_ALPHA, 0)

    return final


# ========================= 视频处理 =========================
def process_video(input_path, output_path, mtx, dist, M, M_inv, src_points):
    """逐帧处理视频并输出"""
    cap    = cv2.VideoCapture(input_path)
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result     = process_single_image(img_rgb, mtx, dist, M, M_inv, src_points)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        out.write(result_bgr)

    cap.release()
    out.release()
    print(f"视频处理完成，保存至 {output_path}")


# ========================= 主程序 =========================
if __name__ == "__main__":
    cal_images = glob.glob("data/camera_cal/calibration*.jpg")
    if cal_images:
        ret, mtx, dist, rvecs, tvecs = cal_calibrate_params(cal_images)
        print("相机标定完成，重投影误差：", ret)
    else:
        mtx  = np.array([[1.0e3, 0, 640], [0, 1.0e3, 360], [0, 0, 1]], dtype=np.float32)
        dist = np.zeros(5)

    test_img  = cv2.imread("data/test_images/straight_lines2.jpg")
    M, M_inv  = cal_perspective_params(test_img, SRC_POINTS)
    print("透视变换矩阵计算完成")

    process_video("data/videos/harder_challenge_video.mp4", "test_output_harder_challenge_video.mp4",
                  mtx, dist, M, M_inv, SRC_POINTS)