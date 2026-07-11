
"""
YOLOv8n-seg 车道线检测（大弯道强化版 v4）

=== 原始问题 ===
  Q1. 大转弯时绿色区域突然消失，平稳后才重新出现
  Q2. 只有单侧车道在画面内时（另一侧出画面），完全检测失败
  Q3. 不确定是代码问题还是模型训练不足
  Q4. 视频转弯多且弯度非常大

=== 修复方案（对应每个问题）===
  FIX-A  [Q2]  extract_lane_centerlines 改为支持单车道返回
                 原版：只要有一侧没检测到就返回 (None, None)
                 新版：能检测到几条就返回几条，没检测到的那一侧返回 None

  FIX-C  [Q1]  自适应平滑（smooth_fit）+ 帧间历史拟合
                 用指数平滑替代 streak 计数，帧间连续性更好

"""

import cv2
import numpy as np
import glob
import os
from ultralytics import YOLO

# ========================= 全局参数 =========================
nx = 9
ny = 6
offset_x = 330
offset_y = 0
xm_per_pix = 3.7 / 700
ym_per_pix = 30 / 720
lane_center_default = 640

SRC_POINTS = [[580, 460], [700, 460], [210, 720], [1110, 720]]
last_left_fit = None
last_right_fit = None

# ========================= 兜底拟合（鸟瞰图坐标，x∈[330,950]）=========================
_FALLBACK_LEFT_FIT  = np.array([0.0, 0.0, 380.0])   # 左侧竖线 x≈380
_FALLBACK_RIGHT_FIT = np.array([0.0, 0.0, 900.0])   # 右侧竖线 x≈900

# ────────────────────────────────────────────────────────────────────────────────

# ========================= YOLO 模型加载 =========================
_model_path = 'runs/segment/runs/yolov8m_lane/weights/best.pt'
if not os.path.exists(_model_path):
    _model_path = 'models/yolov8n-seg.pt'
_model = YOLO(_model_path)


# ========================= YOLO 推理 → 二值图 =========================
def pipeline(img):
    """
    YOLO 分割推理管道：将输入图像转为车道线二值图（0 / 1）。

    处理流程：
      1. YOLOv8n-seg 推理，获取所有分割掩码
      2. 将掩码 resize 到原图尺寸，合并为二值图
      3. 去除上半部分（天空/远景，不包含车道线）
      4. 形态学操作（开运算去噪点 + 闭运算填补空洞）
      5. 高斯模糊 + 二值化（平滑边缘）
      6. 连通域分析，移除小面积噪声（< 300 像素）
    """
    h, w = img.shape[:2]

    results = _model(img, verbose=False, conf=0.25)[0]
    binary = np.zeros((h, w), dtype=np.uint8)
    if results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        for mask in masks:
            m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            binary = cv2.bitwise_or(binary, (m > 0.5).astype(np.uint8))
    binary[:h // 2, :] = 0

    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.GaussianBlur(binary.astype(np.float32), (5, 5), 0)
    binary = (binary > 0.5).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 300:
            binary[labels == i] = 0

    return binary


# ========================= 相机标定与透视变换 =========================

def cal_calibrate_params(file_paths):
    """
    相机标定：从棋盘格照片计算相机内参和畸变系数。

    参数：
        file_paths : list[str]  棋盘格标定图片路径列表

    返回：
        ret   : float  重投影误差（越小越好，通常 < 1.0）
        mtx   : ndarray  相机内参矩阵 (3x3)
        dist  : ndarray  畸变系数 (5,)
        rvecs : list  旋转向量（每张图片）
        tvecs : list  平移向量（每张图片）

    使用 OpenCV 的 findChessboardCorners 检测棋盘格角点，
    棋盘格规格由全局参数 nx x ny 定义（默认为 9x6）。
    """
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
    """用相机内参和畸变系数对图像去畸变。"""
    return cv2.undistort(img, mtx, dist, None, mtx)


def cal_perspective_params(img, src_points):
    """
    计算透视变换矩阵（鸟瞰图 BEV）。

    将四边形 src_points 映射到 dst（一个居中的矩形区域），
    得到鸟瞰图视角，方便对车道线做二次多项式拟合。

    参数：
        img        : ndarray  参考图像（用于获取图像尺寸）
        src_points : list  源四边形四个顶点 [左上, 右上, 左下, 右下]

    返回：
        M     : ndarray  前向透视变换矩阵 (3x3)
        M_inv : ndarray  逆变换矩阵 (3x3)，用于将 BEV 结果映射回原图
    """
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
    """对图像应用透视变换（配合 warpPerspective）。"""
    img_size = (img.shape[1], img.shape[0])
    return cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)


# ========================= 改进后的车道检测函数 =========================

def extract_lane_centerlines(binary, prev_left_fit=None, prev_right_fit=None):
    """
    从二值图中提取左右车道线的中心线点集。

    Bug2 修复：使用「底部 x 坐标排序」+「历史先验最近邻匹配」替代原来的
    mean_x < mid_x(=640) 划分方式，解决弯道中左右车道线偏向同侧时的错配问题。

    原版问题的本质：
      弯道中两条车道线可能都在图像中线的同一侧（例如都 < 640 或都 > 640），
      简单的 mean_x < mid_x 规则会错误地把属于右侧的车道线也归为左侧。

    改进策略：
      1. 用轮廓「底部 20% 区域的平均 x」代替全局 mean_x — 贴地更稳定
      2. 有历史拟合参数时，用预测的底部 x 做最近邻匹配
      3. 无历史时简单地按底部 x 排序：最左→左线，最右→右线
      4. 只有一条轮廓时，根据历史先验或图像中点判断归属

    参数：
        binary        : ndarray  二值图 (h x w)，像素值 0 或 1
        prev_left_fit  : ndarray | None  上一帧左车道线拟合参数 [a, b, c] (y=ax²+bx+c)
        prev_right_fit : ndarray | None  上一帧右车道线拟合参数

    返回：
        (left_centerline, right_centerline) : tuple
            每条中心线为 Nx2 数组，每行 [x, y]；
            未检测到该侧时对应元素为 None。
            返回 (None, None) 表示两侧均未检测到。
    """
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return None, None

    h, w = binary.shape
    # 过滤面积过小的噪声轮廓（提高阈值从100到300）
    valid = [c for c in contours if cv2.contourArea(c) >= 300]
    if not valid:
        return None, None

    def bottom_x(cnt):
        """取轮廓底部20%行的平均x（比全局mean_x更稳定，尤其弯道时）"""
        pts = cnt.reshape(-1, 2)
        thresh = h * 0.80
        bot = pts[pts[:, 1] >= thresh]
        if len(bot) == 0:
            bot = pts[pts[:, 1] >= np.percentile(pts[:, 1], 80)]
        return float(np.mean(bot[:, 0]))

    def make_centerline(cnt):
        """从轮廓提取每行的中心 x"""
        pts = cnt.reshape(-1, 2)
        ys  = np.unique(pts[:, 1])
        center_points = []
        for y in ys:
            row      = pts[pts[:, 1] == y]
            center_x = (np.min(row[:, 0]) + np.max(row[:, 0])) / 2
            center_points.append([center_x, y])
        return np.array(center_points)

    # 按底部 x 升序排列所有有效轮廓（x 越小越靠左）
    sorted_cnts = sorted(valid, key=bottom_x)

    if len(sorted_cnts) == 1:
        # ── 只有一条轮廓：判断它属于左侧还是右侧 ──
        # 有历史拟合时，计算历史左右线在底部的预测 x 位置，
        # 将当前轮廓分配给距离更近的那一侧。
        # 无历史时简单按图像水平中点划分。
        bx = bottom_x(sorted_cnts[0])
        cl = make_centerline(sorted_cnts[0])
        if prev_left_fit is not None and prev_right_fit is not None:
            y_e = h * 0.9
            pl  = prev_left_fit[0]*y_e**2  + prev_left_fit[1]*y_e  + prev_left_fit[2]
            pr  = prev_right_fit[0]*y_e**2 + prev_right_fit[1]*y_e + prev_right_fit[2]
            return (cl, None) if abs(bx - pl) <= abs(bx - pr) else (None, cl)
        else:
            return (cl, None) if bx < w // 2 else (None, cl)

    # 有2+条轮廓
    if prev_left_fit is not None and prev_right_fit is not None:
        # 历史先验：每个轮廓匹配最近的预测线（解决弯道两线偏向同侧的问题）
        y_e = h * 0.9
        pl  = prev_left_fit[0]*y_e**2  + prev_left_fit[1]*y_e  + prev_left_fit[2]
        pr  = prev_right_fit[0]*y_e**2 + prev_right_fit[1]*y_e + prev_right_fit[2]
        left_cnt  = min(sorted_cnts, key=lambda c: abs(bottom_x(c) - pl))
        remaining = [c for c in sorted_cnts if c is not left_cnt]
        right_cnt = min(remaining, key=lambda c: abs(bottom_x(c) - pr))
    else:
        # 无历史：最左->左线，最右->右线
        left_cnt  = sorted_cnts[0]
        right_cnt = sorted_cnts[-1]

    return make_centerline(left_cnt), make_centerline(right_cnt)



def fit_lane_curve(centerline):
    """
    对车道线中心点进行底部加权二次多项式拟合。

    拟合模型：x = a*y² + b*y + c（以 y 为自变量，x 为因变量）
    加权策略：靠近车辆（y 值大）的点权重更高，因为底部区域更可靠。

    参数：
        centerline : ndarray | None  中心线点集 (Nx2)，每行 [x, y]

    返回：
        ndarray | None  拟合系数 [a, b, c]；数据不足（< 30 点）时返回 None
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


def smooth_fit(current_fit, previous_fit, alpha_stable=0.82, alpha_fast=0.45):
    """
    自适应指数平滑：平衡帧间稳定性和响应速度。

    核心思想：
      通过当前帧与历史帧底部位置（常数项 c）的差异大小来判断"这是车道真正在移动
      还是检测噪声"，从而动态选择平滑强度。

    行为：
      变化大（|c_current - c_prev| > 40px）→ alpha_fast（0.45），快速跟随车道变化
      变化小（|c_current - c_prev| <= 40px）→ alpha_stable（0.82），强平滑抑制噪声

    参数：
        current_fit  : ndarray | None  当前帧拟合系数 [a, b, c]
        previous_fit : ndarray | None  上一帧拟合系数
        alpha_stable : float  稳定时的平滑系数（越大越平滑，响应越慢）
        alpha_fast   : float  快速跟随时的平滑系数（越小跟随越快）

    返回：
        ndarray | None  平滑后的拟合系数
    """
    if previous_fit is None:
        return current_fit
    if current_fit is None:
        return previous_fit
    delta = np.abs(current_fit[2] - previous_fit[2])
    alpha = alpha_fast if delta > 40 else alpha_stable
    return alpha * previous_fit + (1 - alpha) * current_fit


def fill_lane_poly(img, left_fit, right_fit, inset=12):
    """
    填充车道区域多边形（在鸟瞰图 BEV 空间）。

    根据左右车道线的拟合曲线计算每行的左右边界，形成闭合多边形后填充绿色。

    参数：
        img      : ndarray  鸟瞰图（仅用于获取尺寸）
        left_fit : ndarray  左车道线拟合系数 [a, b, c]
        right_fit: ndarray  右车道线拟合系数 [a, b, c]
        inset    : int  左右边界向内收缩像素（让绿区不覆盖车道线本身）

    返回：
        ndarray  填充了绿色多边形区域的彩色图像（尺寸同 img）
    """
    out_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    start_y = int(img.shape[0] * 0.54)
    end_y   = img.shape[0]
    w       = img.shape[1]

    left_points  = []
    right_points = []
    for y in range(start_y, end_y):
        lx = left_fit[0]  * y * y + left_fit[1]  * y + left_fit[2]  + inset
        rx = right_fit[0] * y * y + right_fit[1] * y + right_fit[2] - inset

        lx = np.clip(lx, 0, w - 1)
        rx = np.clip(rx, 0, w - 1)
        if lx < rx:
            left_points.append([lx, y])
            right_points.append([rx, y])

    if left_points:
        points = np.vstack((left_points, right_points[::-1]))
        cv2.fillPoly(out_img, [np.int32(points)], (0, 255, 0))

    return out_img


# ========================= 曲率计算 & 车辆偏移 =========================

def cal_radius(img, left_fit, right_fit):
    """
    计算车道曲率半径并在图像上显示。

    将 BEV 空间的像素坐标转换为真实世界坐标（使用 xm_per_pix, ym_per_pix），
    重新拟合后计算曲率半径公式 R = (1 + (2Ay+B)²)^1.5 / |2A|。

    参数：
        img       : ndarray  图像（直接在上面绘制文字）
        left_fit  : ndarray  左车道线 BEV 拟合系数 [a, b, c]
        right_fit : ndarray  右车道线 BEV 拟合系数 [a, b, c]

    返回：
        ndarray  叠加了曲率文字后的图像
    """
    y_vals        = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_x_real   = (left_fit[0]  * y_vals ** 2 + left_fit[1]  * y_vals + left_fit[2])  * xm_per_pix
    right_x_real  = (right_fit[0] * y_vals ** 2 + right_fit[1] * y_vals + right_fit[2]) * xm_per_pix
    y_real        = y_vals * ym_per_pix
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
    """
    计算车辆中心与车道中心的横向偏移量（单位：米）。

    计算方式：取左右车道线底部位置的中点 → 与图像中心比较 → 换算为真实世界偏移。
    结果显示在图像左上角（如 "Vehicle is 0.35 m right of center"）。

    参数：
        img         : ndarray  输入图像（直接在上面绘制文字）
        left_fit    : ndarray  左车道线拟合系数 [a, b, c]
        right_fit   : ndarray  右车道线拟合系数 [a, b, c]
        lane_center : int  图像中心 x 坐标（用于计算偏移，默认 640）

    返回：
        ndarray  叠加了偏移量文字后的图像
    """
    y_max             = img.shape[0] - 1
    left_x            = left_fit[0]  * y_max ** 2 + left_fit[1]  * y_max + left_fit[2]
    right_x           = right_fit[0] * y_max ** 2 + right_fit[1] * y_max + right_fit[2]
    lane_center_pixel = (left_x + right_x) / 2.0
    offset_m          = (lane_center_pixel - lane_center) * xm_per_pix
    if offset_m > 0:
        text = f'Vehicle is {offset_m:.2f} m right of center'
    elif offset_m < 0:
        text = f'Vehicle is {-offset_m:.2f} m left of center'
    else:
        text = 'Vehicle is in the center'
    cv2.putText(img, text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img



# ========================= 单帧处理核心 =========================
def process_single_image(img, mtx, dist, M, M_inv, src_points):
    """
    处理单帧图像：去畸变 → YOLO 分割 → 透视变换 → 车道线提取拟合
    → 质量判定/平滑/验证 → 渲染车道区域 → 叠加诊断信息。

    这是整个系统的核心编排函数，串联所有子模块。
    使用帧间历史（last_left_fit, last_right_fit）实现平滑和稳定性。

    参数：
        img        : ndarray  输入 RGB 图像
        mtx        : ndarray  相机内参矩阵
        dist       : ndarray  畸变系数
        M          : ndarray  透视变换矩阵（前向）
        M_inv      : ndarray  透视变换矩阵（逆向）
        src_points : list  透视变换源四边形顶点

    返回：
        ndarray  处理后的图像（含车道区域叠加和诊断信息）
    """
    global last_left_fit, last_right_fit

    undist = img_undistort(img, mtx, dist)
    binary = pipeline(undist)
    warp   = img_perspect_transform(binary, M)

    # ── 传入历史先验解决弯道左右错配 ───────────────────────────────
    # Bug2 修复：将上一帧的左右拟合参数传给 extract_lane_centerlines，
    # 使其在弯道两侧线偏向同侧时能通过最近邻匹配正确分配左右。
    left_lane, right_lane = extract_lane_centerlines(
        warp,
        prev_left_fit=last_left_fit,
        prev_right_fit=last_right_fit
    )
    raw_left  = fit_lane_curve(left_lane)   # 拟合左车道线：x = a*y² + b*y + c
    raw_right = fit_lane_curve(right_lane)  # 拟合右车道线

    # ═══════════════ 检测结果分类 ═══════════════════════════════
    #
    # 检测状态：
    #   2 = 双侧完整检测
    #   0 = 未检测到或只有单侧（单侧结果不可靠，跳过此帧）
    # ═══════════════════════════════════════════════════════════

    if raw_left is not None and raw_right is not None:
        quality = 2
    else:
        quality = 0

    # ═══════════════ 帧间状态更新 ═══════════════════════════════
    # （简化：不再跟踪 streak / no_detect_frames，所有检测直接渲染）
    # ═════════════════════════════════════════════════════════════════════

    # ── None 保护：给平滑函数提供安全的 fallback ──
    # 当某侧检测失败时用历史数据或兜底值替代，确保平滑计算不会因 None 而中断。
    left_fit  = (raw_left  if raw_left  is not None
                 else (last_left_fit  if last_left_fit  is not None else _FALLBACK_LEFT_FIT.copy()))
    right_fit = (raw_right if raw_right is not None
                 else (last_right_fit if last_right_fit is not None else _FALLBACK_RIGHT_FIT.copy()))
    # ─────────────────────────────────────────────────────────────────

    # ── 自适应平滑（过滤检测噪声，同时保留车道真实变化）───────
    # 仅当 quality>=1（有检测结果）时才更新 last_fit，防止错误结果污染历史。
    smoothed_left  = smooth_fit(left_fit,  last_left_fit)
    smoothed_right = smooth_fit(right_fit, last_right_fit)
    if quality >= 1:
        last_left_fit  = smoothed_left
        last_right_fit = smoothed_right
    # 更新最终使用的拟合参数（如有历史则使用历史，否则用兜底值）
    left_fit  = last_left_fit  if last_left_fit  is not None else _FALLBACK_LEFT_FIT.copy()
    right_fit = last_right_fit if last_right_fit is not None else _FALLBACK_RIGHT_FIT.copy()
    # ─────────────────────────────────────────────────────────────────────────

    # ── 渲染车道区域 ──
    # 在 BEV 空间填充绿色多边形，
    # 逆透视变换回原图视角，叠加曲率半径和车辆偏移文字，
    # 按固定透明度（0.5）与去畸变原图融合。
    warp_color = fill_lane_poly(warp, left_fit, right_fit)
    inv_warp   = img_perspect_transform(warp_color, M_inv)
    result     = cal_radius(inv_warp, left_fit, right_fit)
    result     = cal_center_departure(result, left_fit, right_fit, lane_center=lane_center_default)
    final      = cv2.addWeighted(undist, 1, result, 0.5, 0)

    return final


# ========================= 视频处理（逐帧处理 + 输出）=================
def process_video(input_path, output_path, mtx, dist, M, M_inv, src_points):
    """
    读取输入视频，逐帧调用 process_single_image 处理，写入输出视频。

    每 30 帧在控制台打印一次进度。

    参数：
        input_path  : str  输入视频路径（如 data/videos/harder_challenge_video.mp4）
        output_path : str  输出视频路径（如 output_yolo_video.mp4）
        mtx, dist, M, M_inv, src_points : 相机标定 + 透视变换参数
                                          由主程序计算后传入
    """
    cap    = cv2.VideoCapture(input_path)
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result     = process_single_image(img_rgb, mtx, dist, M, M_inv, src_points)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        out.write(result_bgr)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  已处理 {frame_idx} 帧...")
    cap.release()
    out.release()
    print(f"✓ 视频处理完成，保存至 {output_path}（共 {frame_idx} 帧）")


# ========================= 主程序（标定 + 透视 + 处理视频）============
if __name__ == "__main__":
    # ── 第 1 步：相机标定（从棋盘格图片计算内参和畸变）──
    cal_images = glob.glob("data/camera_cal/calibration*.jpg")
    if cal_images:
        ret, mtx, dist, rvecs, tvecs = cal_calibrate_params(cal_images)
        print("✓ 相机标定完成，重投影误差：", ret)
    else:
        print("⚠ 未找到标定图片，使用默认相机矩阵（可能不够精确）")
        mtx  = np.array([[1.0e3, 0, 640], [0, 1.0e3, 360], [0, 0, 1]], dtype=np.float32)
        dist = np.zeros(5)

    # ── 第 2 步：计算透视变换矩阵（前向 + 逆变换）──
    test_img = cv2.imread("data/test_images/straight_lines2.jpg")
    M, M_inv = cal_perspective_params(test_img, SRC_POINTS)
    print("✓ 透视变换矩阵计算完成")

    # ── 第 3 步：处理视频 ──
    process_video(
        "data/videos/harder_challenge_video.mp4",
        "output/output_harder_challenge_video.mp4",
        mtx, dist, M, M_inv, SRC_POINTS
    )
    # process_video(
    #     "data/videos/challenge_video.mp4",
    #      "output/output_challenge_video.mp4",
    #     mtx, dist, M, M_inv, SRC_POINTS
    # )
    # process_video(
    #     "data/videos/project_video.mp4",
    #      "output/output_project_video.mp4",
    #     mtx, dist, M, M_inv, SRC_POINTS
    # )