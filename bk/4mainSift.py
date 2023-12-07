import cv2
import numpy as np
import os
from scipy.spatial import Delaunay

# 图像文件夹路径和PLY文件保存路径
img_dir = "C:/Users/crab/PycharmProjects/3dReset/images/"
ply_save_path = "C:/Users/crab/PycharmProjects/3dReset/results/result.ply"
depth_image_file = "depth_image.png"

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
for i in range(5):  # 图片范围从0000.jpg到0004.jpg
    for j in range(i+1, 5):  # 确保j不超过最后一张图片的索引
        img_path1 = os.path.join(img_dir, f"{i:04d}.jpg")
        img_path2 = os.path.join(img_dir, f"{j:04d}.jpg")

        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)
        if img1 is None or img2 is None:
            print(f"Warning: Unable to read one of the images: {img_path1} or {img_path2}")
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

        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16 * 3,  # 这个值应是16的倍数
            blockSize=5,
            P1=8 * 3 * 5 ** 2,
            P2=32 * 3 * 5 ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )

        # 计算视差图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        disparity = stereo.compute(gray1, gray2).astype(np.float32) / 16.0

        # 从视差图计算深度图
        f = K[0, 0]  # 相机的焦距
        T = np.linalg.norm(t)  # 从R,t中获取相机基线距离
        Q = np.float32([[1, 0, 0, -0.5 * K[0, 2]],
                        [0, -1, 0, 0.5 * K[1, 2]],
                        [0, 0, 0, -f],
                        [0, 0, 1 / T, 0]])
        points_3D = cv2.reprojectImageTo3D(disparity, Q)

        # 处理生成的稠密点云
        valid_points = (disparity > disparity.min()) & (points_3D[:, :, 2] < some_max_depth)  # 根据需要设置深度阈值
        dense_point_cloud = points_3D[valid_points]
        dense_colors = img1[valid_points]  # 假设img1是参考图像

        # 将稠密点云添加到总点云中
        point_cloud = np.vstack((point_cloud, dense_point_cloud.reshape(-1, 3)))
        point_colors.extend(dense_colors)
        # 可以选择显示视差图和深度图
        # cv2.imshow('Disparity', (disparity - min_disp) / num_disp)
        # cv2.imshow('Depth Map', points_3D[:, :, 2])
        # cv2.waitKey(0)


        # 添加点到点云
        point_cloud = np.vstack((point_cloud, point3D.reshape(-1, 3)))

        # 获取特征点的颜色信息
        for pt in inlierPoints1:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < img1.shape[1] and 0 <= y < img1.shape[0]:
                point_colors.append(img1[y, x])

# 确保点云和颜色数据长度一致
assert len(point_cloud) == len(point_colors), "点云和颜色数据长度不一致"


##有时间再
# 读取点云
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(point_cloud)
# pcd.colors = o3d.utility.Vector3dVector(point_colors)
#
# # 去噪
# cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
# pcd = pcd.select_by_index(ind)
#
# # 下采样
# voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)

# 平滑处理（可选）
# 这里需要一些额外的步骤，例如估计法线

# 使用Delaunay三角化来生成三维面片
if len(point_cloud) > 3:
    tri = Delaunay(point_cloud)
    meshes = point_cloud[tri.simplices]
else:
    meshes = []

# 面片扩张和过滤的迭代
# 定义面片扩张的阈值
expansion_threshold = 0.1  # 根据需要调整阈值

# 定义面片过滤的阈值
filter_threshold = 0.2  # 根据需要调整阈值

# 初始化新的三维面片列表
new_meshes = []

# 遍历现有的面片列表
for triangle in meshes:
    # 计算当前面片的中心点
    triangle_center = np.mean(triangle, axis=0)

    # 寻找与中心点距离在阈值内的稀疏点云点
    nearby_points = [point for point in point_cloud if np.linalg.norm(point - triangle_center) < expansion_threshold]

    # 如果找到附近的点，则合并当前面片的点云和附近的稀疏点云点
    if nearby_points:
        expanded_triangle = np.vstack((triangle, nearby_points))
        # ... 进行面片过滤


        # 过滤掉不满足条件的面片
        if len(expanded_triangle) >= 3:
            # 计算新面片的法向量
            normal_vector = np.cross(expanded_triangle[1] - expanded_triangle[0], expanded_triangle[2] - expanded_triangle[0])
            normal_vector /= np.linalg.norm(normal_vector)

            # 根据法向量和阈值进行面片过滤
            if np.abs(np.dot(normal_vector, [0, 0, 1])) > filter_threshold:
                new_meshes.append(expanded_triangle)
    else:
        # 如果没有找到附近的点，可以选择跳过当前面片或直接添加到 new_meshes
        continue  # 或 new_meshes.append(triangle)

# 深度图像的生成
image_height, image_width = img1.shape[:2]  # 使用第一张图像的尺寸
depth_image_size = (image_height, image_width)
initial_depth_value = 1000.0  # 初始深度值

# 创建初始深度图像
depth_image = np.ones(depth_image_size, dtype=np.float32) * initial_depth_value

# 遍历三维面片列表new_meshes
for triangle in new_meshes:
    # 计算当前三维面片的中心点
    triangle_center = np.mean(triangle, axis=0)

    # 计算三维面片的投影坐标
    projected_points, _ = cv2.projectPoints(triangle, np.zeros(3), np.zeros(3), K, None)

    # 将投影坐标转换为图像坐标
    projected_points = projected_points.squeeze().astype(int)

    # 将深度值写入深度图像
    for i, point in enumerate(projected_points):
        x, y = point
        if 0 <= x < depth_image_size[1] and 0 <= y < depth_image_size[0]:
            current_depth = np.linalg.norm(triangle_center - triangle[i])
            if current_depth < depth_image[y, x]:
                depth_image[y, x] = current_depth

# 保存深度图像为PNG文件
cv2.imwrite(depth_image_file, depth_image)
print(f"Depth image saved to {depth_image_file}")
