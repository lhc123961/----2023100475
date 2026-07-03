import cv2
import numpy as np
import glob

# ================== 棋盘格参数设置 ==================
# 内角点数量：13列 × 13行
pattern_size = (9, 6)
square_size = 25.0             # 方格边长 1cm = 10mm

# 准备标定板在世界坐标系中的三维坐标（z=0 平面）
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

# 存储三维点与二维图像点
obj_points = []   # 世界坐标
img_points = []   # 图像坐标

# ================== 读取标定图片 ==================
images = glob.glob('calib_images/*.jpg')
if len(images) == 0:
    print("错误：在 calib_images 文件夹中未找到任何 .jpg 图片，请检查路径。")
    exit()

print(f"共找到 {len(images)} 张图片，开始角点检测...")

successful_images = []   # 保存成功检测的原图路径（用于后续去畸变）
image_size = None        # 图像尺寸 (宽度, 高度)

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"无法读取 {fname}，跳过")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 记录图像尺寸（所有图片应尺寸相同）
    if image_size is None:
        image_size = gray.shape[::-1]  # (width, height)
    else:
        if gray.shape[::-1] != image_size:
            print(f"警告：{fname} 尺寸不一致，可能影响标定结果。")

    # 查找棋盘格内角点
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        # 亚像素精度优化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_sub = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

        obj_points.append(objp)
        img_points.append(corners_sub)

        # 绘制角点并保存预览图
        drawn_img = cv2.drawChessboardCorners(img, pattern_size, corners_sub, ret)
        out_name = fname.replace('.jpg', '_corners.jpg').replace('calib_images/', 'calib_images/corners_')
        cv2.imwrite(out_name, drawn_img)
        successful_images.append(fname)   # 保存原图路径，方便后续使用

        print(f"√ {fname} 检测成功，角点图已保存")
    else:
        print(f"× {fname} 未检测到完整棋盘格，跳过")

print(f"\n成功检测角点的图片数量：{len(obj_points)}")
if len(obj_points) < 15:
    print("提示：有效图片少于15张，建议增加不同姿态的图片以提高标定精度。")

# ================== 相机标定 ==================
print("\n开始相机标定...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, image_size, None, None
)

print("\n========== 标定结果 ==========")
print("内参矩阵 K：")
print(mtx)
print("\n畸变系数 [k1, k2, p1, p2, k3]：")
print(dist.ravel())
print(f"\n总体重投影误差（RMS）：{ret:.4f} 像素")

# 计算每张图片的平均重投影误差
mean_errors = []
for i in range(len(obj_points)):
    projected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(img_points[i], projected, cv2.NORM_L2) / len(projected)
    mean_errors.append(error)

total_error = np.mean(mean_errors)
print(f"所有图片的平均重投影误差：{total_error:.4f} 像素")

# ================== 去畸变对比 ==================
print("\n生成去畸变对比图...")
if len(successful_images) > 0:
    demo_img_path = successful_images[0]   # 使用第一张成功检测的原图
    demo_img = cv2.imread(demo_img_path)
    if demo_img is not None:
        # 去畸变
        undistorted_img = cv2.undistort(demo_img, mtx, dist, None, mtx)

        # 水平拼接原图和去畸变图（左：原图，右：去畸变）
        comparison = np.hstack((demo_img, undistorted_img))
        cv2.imwrite('undistortion_comparison.png', comparison)
        print("去畸变对比图已保存为 undistortion_comparison.png （左：原图，右：去畸变）")

        # 尝试显示对比图（如果运行环境支持 GUI）
        try:
            cv2.imshow('Original (left) vs Undistorted (right)', comparison)
            print("按任意键关闭图像窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            pass
    else:
        print("无法读取演示原图，跳过去畸变对比。")
else:
    print("没有可用的图片进行去畸变对比。")

# ================== 简要分析 ==================
print("\n========== 简要分析 ==========")
fx = mtx[0, 0]
fy = mtx[1, 1]
cx = mtx[0, 2]
cy = mtx[1, 2]

print(f"fx = {fx:.2f}, fy = {fy:.2f}")
print(f"cx = {cx:.2f}, cy = {cy:.2f}")
if image_size is not None:
    center_x = image_size[0] / 2
    center_y = image_size[1] / 2
    print(f"图像中心坐标：({center_x:.2f}, {center_y:.2f})")

    if abs(fx - fy) / fx < 0.05:
        print("fx 与 fy 非常接近，像素形状接近正方形。")
    else:
        print("fx 与 fy 差异较大，像素可能非正方形或镜头存在变形。")

    if abs(cx - center_x) < image_size[0] * 0.05 and abs(cy - center_y) < image_size[1] * 0.05:
        print("主点 (cx, cy) 接近图像中心，光轴与传感器中心偏移较小。")
    else:
        print("主点偏离图像中心，可能存在镜头与传感器的安装偏移。")
else:
    print("未获得图像尺寸，无法分析主点偏移。")

if total_error < 1.0:
    print(f"重投影误差 {total_error:.4f} 像素，标定精度较高。")
elif total_error < 2.0:
    print(f"重投影误差 {total_error:.4f} 像素，结果可接受。")
else:
    print(f"重投影误差 {total_error:.4f} 像素，偏大。建议检查棋盘格平整度、图像清晰度，或增加更多姿态的图片。")

print("\n程序执行完毕。")