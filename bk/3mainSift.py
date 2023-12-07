import cv2
import numpy as np
import os

# 图像文件夹路径和PLY文件保存路径
img_dir = "C:/Users/crab/PycharmProjects/3dReset/castle-P19/images/"
ply_save_path = "C:/Users/crab/PycharmProjects/3dReset/results/result.ply"

# 相机内参矩阵K
K = np.array([[2759.48, 0, 1520.69],
              [0, 2764.16, 1006.81],
              [0, 0, 1]])

# 创建SIFT对象
sift = cv2.SIFT_create()

# 初始化点云和颜色数组
point_cloud = np.empty((0, 3))
point_colors = []

# 处理所有图片对
for i in range(18):  # 假设有18张图片
    for j in range(i+1, 7):  # 与后面的每张图像进行匹配 #!!!!最右边的数字参数的修改  决定了它需要跟多少个人进行匹配
        img_path1 = os.path.join(img_dir, f"{i:04d}.jpg")
        img_path2 = os.path.join(img_dir, f"{j:04d}.jpg")

        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)
        if img1 is None or img2 is None:
            continue

        # 检测关键点并计算描述子
        keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

        # 使用FLANN进行特征匹配
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        # 筛选好的匹配
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

        # 保存匹配点对坐标
        matchedPoints1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
        matchedPoints2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

        # 进行基本矩阵F的估计并使用RANSAC筛选
        F, inliers = cv2.findFundamentalMat(matchedPoints1, matchedPoints2, cv2.FM_RANSAC)

        # 筛选内点
        inlierPoints1 = matchedPoints1[inliers.ravel() == 1]
        inlierPoints2 = matchedPoints2[inliers.ravel() == 1]

        # 计算本质矩阵E
        E = K.T @ F @ K

        # 相机姿态恢复，求解R,t
        _, R, t, _ = cv2.recoverPose(E, inlierPoints1, inlierPoints2, K)

        # 创建两个相机的投影矩阵 [R T]
        proj1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
        proj2 = np.hstack((R, t))

        # 转换相机内参矩阵 K 为浮点型
        fK = K.astype(np.float32)

        # 计算投影矩阵 [K * [R|T]]
        proj1 = fK @ proj1
        proj2 = fK @ proj2

        # 三角法求解稀疏三维点云
        point4D_homogeneous = cv2.triangulatePoints(proj1, proj2, inlierPoints1.T, inlierPoints2.T)
        point3D = cv2.convertPointsFromHomogeneous(point4D_homogeneous.T)

        # 添加点到点云
        point_cloud = np.vstack((point_cloud, point3D.reshape(-1, 3)))

        # 获取特征点的颜色信息
        for pt in inlierPoints1:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < img1.shape[1] and 0 <= y < img1.shape[0]:
                point_colors.append(img1[y, x])

# 确保点云和颜色数据长度一致
assert len(point_cloud) == len(point_colors), "点云和颜色数据长度不一致"

# 手动输出点云ply文件
with open(ply_save_path, 'w') as plyFile:
    plyFile.write("ply\n")
    plyFile.write("format ascii 1.0\n")
    plyFile.write("element vertex {}\n".format(len(point_cloud)))
    plyFile.write("property float x\n")
    plyFile.write("property float y\n")
    plyFile.write("property float z\n")
    plyFile.write("property uchar red\n")
    plyFile.write("property uchar green\n")
    plyFile.write("property uchar blue\n")
    plyFile.write("end_header\n")
    for point, color in zip(point_cloud, point_colors):
        plyFile.write("{} {} {} {} {} {}\n".format(point[0], point[1], point[2], color[2], color[1], color[0]))

print("PLY file saved to", ply_save_path)
