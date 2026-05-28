import cv2
import numpy as np

# --- 0. 基础设置与图片读取 ---
path1 = r"C:\war3\test1.png"
path2 = r"C:\war3\test2.png"

# 读取为灰度图（用于特征检测）
img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

# 读取彩色图（用于最后画框显示）
img2_color = cv2.imread(path2)

# 检查图片是否读取成功
if img1 is None or img2 is None:
    print("错误：无法读取图片，请检查路径是否正确！")
    exit()

print("图片读取成功，开始执行任务...")

# --- 任务 1: ORB 特征检测 ---
print("--- 任务 1: ORB 特征检测 ---")
# 1. 创建 ORB 对象，设置最大特征点数量为 1000
orb = cv2.ORB_create(nfeatures=1000)

# 2. 检测关键点并计算描述符
# kp: 关键点 (KeyPoints), des: 描述符 (Descriptors)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

print(f"在图1中检测到 {len(kp1)} 个特征点")
print(f"在图2中检测到 {len(kp2)} 个特征点")

# 绘制关键点图像
img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0))
img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0))

# 显示 ORB 结果
cv2.imshow('ORB Keypoints - Image 1', img1_kp)
cv2.imshow('ORB Keypoints - Image 2', img2_kp)
cv2.waitKey(0)  # 按任意键继续


# --- 任务 2: ORB 特征匹配 ---
print("--- 任务 2: ORB 特征匹配 ---")
# 1. 创建 BFMatcher 对象
# cv2.NORM_HAMMING: 因为 ORB 是二进制描述符，所以使用汉明距离
# crossCheck=False: 这里先设为False，以便使用 knnMatch 进行比率测试（效果更好）
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# 2. 使用 k-近邻匹配 (k=2)，我们需要前两个最佳匹配来计算比率
matches = bf.knnMatch(des1, des2, k=2)

# 3. 应用比率测试 (Ratio Test) 筛选好的匹配点
# 这是作业中提到的 "筛选匹配对"
good_matches = []
for m, n in matches:
    # 如果最佳匹配的距离远小于次佳匹配（例如小于 0.75），则认为它是可靠的
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f"经过比率测试后，保留了 {len(good_matches)} 个良好匹配点")

# 绘制前 30 个匹配点用于查看（或者所有良好匹配点）
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:30], None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('Good Matches', img_matches)
cv2.waitKey(0)


# --- 任务 3: RANSAC 剔除错误匹配 ---
print("--- 任务 3: RANSAC 剔除错误匹配 ---")
# 要计算单应性矩阵，我们需要至少 4 对点
if len(good_matches) > 4:
    # 1. 提取匹配点的坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 2. 使用 RANSAC 计算单应性矩阵 (Homography)
    # M: 单应性矩阵, mask: 掩膜（标记哪些点是内点/正确匹配）
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 获取内点的数量
    matches_mask = mask.ravel().tolist()
    inliers = sum(matches_mask)
    print(f"RANSAC 筛选后，内点（正确匹配）数量为: {inliers}")

    # --- 任务 4: 目标定位 ---
    print("--- 任务 4: 目标定位 ---")
    # 1. 获取模板图片（小图）的四个角点坐标
    h, w = img1.shape
    # 定义四个角：左上，右上，右下，左下
    pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

    # 2. 使用透视变换将四个角投影到场景图（大图）中
    if M is not None:
        dst = cv2.perspectiveTransform(pts, M)

        # 3. 在场景图上画出矩形框
        # 将点转换为整数以便绘图
        pts_dst = np.int32(dst)

        # 使用 polylines 画闭合的多边形（矩形）
        img2_final = cv2.polylines(img2_color, [pts_dst], True, (0, 255, 0), 3, cv2.LINE_AA)

        # 4. 显示结果
        print("定位完成，正在显示最终结果...")
        cv2.imshow('Object Detection Result', img2_final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("未能计算出单应性矩阵 M")

else:
    print(f"匹配点不足 4 个（只有 {len(good_matches)} 个），无法计算单应性矩阵。")
    cv2.destroyAllWindows()
cv2.waitKey(0)

# 销毁所有创建的窗口
cv2.destroyAllWindows()

 # --- 任务 6: 参数对比实验 ---
print("--- 任务 6: 目标定位 ---")
 #1. 图片路径设置

template_path = path1  
scene_path = path2     
# 读取图片 (灰度图用于处理，彩色图用于显示)
img1 = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(scene_path, cv2.IMREAD_GRAYSCALE)
img2_color = cv2.imread(scene_path)  # 用于最后画框的彩色图

# 检查图片是否加载成功
if img1 is None or img2 is None:
    print("错误：图片读取失败，请检查路径。")
    exit()

print(f"模板图尺寸: {img1.shape}, 场景图尺寸: {img2.shape}")
print("-" * 80)

# 2. 定义要测试的参数列表
nfeatures_list = [500, 1000, 2000]

# 打印表格头
print(f"{'nfeatures':<10} {'模板关键点':<12} {'场景关键点':<12} {'匹配数':<10} {'RANSAC内点数':<12} {'内点比例':<10} {'是否定位'}")
print("-" * 80)

# 3. 循环测试不同的 nfeatures
for nfeat in nfeatures_list:
    # --- 初始化 ORB 检测器 ---
    orb = cv2.ORB_create(nfeatures=nfeat)

    # --- 检测关键点和计算描述子 ---
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # --- 特征匹配 (BFMatcher) ---
    # ORB 使用汉明距离 (NORM_HAMMING)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # 使用 kNN 匹配，k=2 用于后续的 Lowe's Ratio Test
    matches = bf.knnMatch(des1, des2, k=2)

    # --- Lowe's Ratio Test 筛选匹配点 ---
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # --- RANSAC 剔除误匹配并计算单应性矩阵 ---
    is_success = False
    inliers_count = 0
    ratio_inliers = 0.0

    # 只有当好的匹配点足够多（至少4个）时，才进行透视变换计算
    if len(good_matches) >= 4:
        # 提取匹配点的坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 计算单应性矩阵 (Homography) 和 掩膜 (Mask)
        # 掩膜中为 1 的是内点 (Inliers)，为 0 的是外点 (Outliers)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:
            is_success = True
            inliers_count = int(mask.sum())
            ratio_inliers = inliers_count / len(good_matches)

           
            if nfeat == 2000:
                h, w = img1.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                img2_result = cv2.polylines(img2_color.copy(), [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.imshow(f'Result nfeatures={nfeat}', img2_result)

    # --- 打印当前参数的实验数据 ---
    status = "是" if is_success else "否"
    print(
        f"{nfeat:<10} {len(kp1):<12} {len(kp2):<12} {len(good_matches):<10} {inliers_count:<12} {ratio_inliers:.2f}       {status}"
    )

print("-" * 80)
print("实验结束。按任意键关闭所有窗口...")
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import numpy as np

# 1. 读取图片
# 请先定义好路径（注意 r 代表原样输出，防止 \ 转义）
path1 = r"C:\war3\test1.png"
path2 = r"C:\war3\test2.png"

# 然后用 imread 读取图片数据
# 注意：这里不需要加引号包裹变量名
img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
img2_color = cv2.imread(path2)                  # 读取为彩色图（用于最后画框）


# 检查图片是否读取成功
if img1 is None or img2 is None:
    print("错误：图片未找到，请检查文件名。")
    exit()

# 2. 创建 SIFT 检测器
# SIFT 是免费的算法，直接创建即可
# 注意：这里不再是 cv2.ORB_create() 了
sift = cv2.SIFT_create()

# 3. 检测关键点并计算描述子
# detectAndCompute 函数会同时做这两件事
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

print(f"SIFT 检测到模板图关键点数量: {len(kp1)}")
print(f"SIFT 检测到场景图关键点数量: {len(kp2)}")

# 4. 创建匹配器
# 注意：SIFT 的描述子是浮点数，所以不能用汉明距离
# 必须使用 cv2.NORM_L2
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# 5. 进行匹配 (使用 knnMatch 进行 k=2 的最近邻匹配，用于比例测试)
matches = bf.knnMatch(des1, des2, k=2)

# 6. 筛选匹配点 (Lowe's Ratio Test)
# SIFT 通常使用 0.75 或 0.8 作为阈值，比 ORB 的 0.75 稍微宽松一点也可以
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f"筛选后的良好匹配点数量: {len(good_matches)}")

# 7. 目标定位 (RANSAC)
# 只有当匹配点足够多时（比如大于 4 个），才进行透视变换
if len(good_matches) >= 4:
    # 获取匹配点的坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 计算单应性矩阵 (Homography)
    # 使用 RANSAC 算法去除误匹配
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 统计内点数量
    matches_mask = mask.ravel().tolist()
    h, w = img1.shape

    # 计算内点比例
    inliers = sum(matches_mask)
    ratio = inliers / len(good_matches) if len(good_matches) > 0 else 0
    print(f"RANSAC 内点数量: {inliers}, 内点比例: {ratio:.2f}")

    # 如果内点足够多，则画出边框
    if inliers > 10:
        # 定义模板图的四个角
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        # 投影到场景图中
        dst = cv2.perspectiveTransform(pts, M)

        # 在场景图中画线
        img2_color = cv2.polylines(img2_color, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
        print(">>> 目标定位成功！")
    else:
        print(">>> 内点不足，定位失败。")
        matches_mask = None
else:
    print(f"匹配点太少 ({len(good_matches)}/4)，无法计算变换矩阵。")
    matches_mask = None

# 8. 绘制结果
# drawMatchesKnn 需要 matches 是二维列表，但我们要画的是筛选后的一维 good_matches
# 所以这里直接用 drawMatches
draw_params = dict(
    matchColor=(0, 255, 0),  # 绿色线条
    singlePointColor=None,
    matchesMask=matches_mask, # 只画内点
    flags=2
)

img_result = cv2.drawMatches(img1, kp1, img2_color, kp2, good_matches, None, **draw_params)

# 显示结果
cv2.imshow('SIFT 匹配结果', img_result)
print("显示结果窗口...")
cv2.waitKey(0)
cv2.destroyAllWindows()