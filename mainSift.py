import cv2
import numpy as np
import os
from scipy.spatial import Delaunay
import open3d as o3d
import cv2
import numpy as np
import os
import scipy
from scipy.spatial.distance import cosine


# 定义一个函数来找到全局点云中的点与新图像中的特征点之间的匹配
# 定义一个函数来找到全局点云中的点与新图像中的特征点之间的匹配  3:128维度的匹配， 我现在需要做的是， 进行系统的知识学习
def find_matches(global_point_cloud, keypoints_new, descriptors_new):
    matched_3d_points = []  # 存储匹配的全局点云中的点
    matched_2d_points = []  # 存储匹配的新图像中的特征点

    # 遍历新图像中的关键点和描述子
    for i, keypoint in enumerate(keypoints_new):
        descriptor = descriptors_new[i][:128]  # 限制描述子的长度为128以匹配点的维度

        # 初始化最小距离和最近的点
        min_distance = float('inf')
        closest_point = None

        # 遍历全局点云中的点
        for point in global_point_cloud:
            # 计算描述子之间的距离（使用余弦相似度作为距离度量）
            distance = cosine(descriptor, point[:128])

            # 如果找到更近的点，则更新最小距离和最近的点
            if distance < min_distance:
                min_distance = distance
                closest_point = point

        # 如果找到了匹配点，则将它们添加到匹配列表中
        if closest_point is not None:
            matched_3d_points.append(closest_point)
            matched_2d_points.append(keypoint.pt)

    return np.array(matched_3d_points), np.array(matched_2d_points)


# 剩余部分保持不变

# Bundle Adjustment (BA) 优化
# 定义BA优化函数
def bundle_adjustment(camera_poses, point_cloud, K, max_iterations=100):
    num_cameras = len(camera_poses)
    num_points = len(point_cloud)

    # 构建优化问题
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iterations, 0.1)
    flags = cv2.OPTIMIZER_SCHUR
    optimizer = cv2.UMat(cv2.detail.createBundleAdjuster())

    # 将相机参数和三维点云连接成一个优化问题的参数向量
    params = np.zeros((3 * num_cameras + 3 * num_points), dtype=np.float32)

    # 初始化优化问题的参数向量
    for i in range(num_cameras):
        params[3 * i:3 * i + 3] = cv2.Rodrigues(camera_poses[i][:3, :3])[0]
        params[3 * num_cameras + 3 * i:3 * num_cameras + 3 * i + 3] = point_cloud[i]

    # 优化相机参数和三维点云
    (success, params) = optimizer.apply(params, K, criteria, flags)

    # 更新相机姿态和三维点云
    for i in range(num_cameras):
        camera_poses[i][:3, :3] = cv2.Rodrigues(params[3 * i:3 * i + 3])[0]
        point_cloud[i] = params[3 * num_cameras + 3 * i:3 * num_cameras + 3 * i + 3]

    return camera_poses, point_cloud

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
global_point_cloud = np.empty((0, 3))
global_camera_poses = []
point_colors = []
keypoints_all = []

# 选择初始视图 a 和 b
init_view_a = 0
init_view_b = 1

# 选择初始视图的图像路径
init_img_path_a = os.path.join(img_dir, f"{init_view_a:04d}.jpg")
init_img_path_b = os.path.join(img_dir, f"{init_view_b:04d}.jpg")

img_a = cv2.imread(init_img_path_a)
img_b = cv2.imread(init_img_path_b)

# 检测关键点并计算描述子
keypoints_a, descriptors_a = sift.detectAndCompute(img_a, None)
keypoints_b, descriptors_b = sift.detectAndCompute(img_b, None)

#加入关键点
keypoints_all.append(keypoints_a)
keypoints_all.append(keypoints_b)

# 使用FLANN进行特征匹配
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors_a, descriptors_b, k=2)

# 筛选好的匹配
good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

# 保存匹配点对坐标
matchedPoints_a = np.float32([keypoints_a[m.queryIdx].pt for m in good_matches])
matchedPoints_b = np.float32([keypoints_b[m.trainIdx].pt for m in good_matches])

# 进行基本矩阵F的估计并使用RANSAC筛选
F, inliers = cv2.findFundamentalMat(matchedPoints_a, matchedPoints_b, cv2.FM_RANSAC)

# 筛选内点
inlierPoints_a = matchedPoints_a[inliers.ravel() == 1]
inlierPoints_b = matchedPoints_b[inliers.ravel() == 1]

# 计算本质矩阵E
E = K.T @ F @ K

# 相机姿态恢复，求解R,t
_, R_init, t_init, _ = cv2.recoverPose(E, inlierPoints_a, inlierPoints_b, K)

# 三角法求解稀疏三维点云
proj1_init = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
proj2_init = np.hstack((R_init, t_init))

point4D_homogeneous_init = cv2.triangulatePoints(proj1_init, proj2_init, inlierPoints_a.T, inlierPoints_b.T)
point3D_init = cv2.convertPointsFromHomogeneous(point4D_homogeneous_init.T)

# 添加初始视图的点云到全局点云
global_point_cloud = np.vstack((global_point_cloud, point3D_init.reshape(-1, 3)))

# 对后续视图进行处理
for i in range(2, 5):  # 处理0002.jpg到0004.jpg
    new_img_path = os.path.join(img_dir, f"{i:04d}.jpg")
    new_img = cv2.imread(new_img_path)

    if new_img is None:
        print(f"Warning: Unable to read the image: {new_img_path}")
        continue

    # 检测关键点并计算描述子
    keypoints_new, descriptors_new = sift.detectAndCompute(new_img, None)

    # 寻找与全局点云中点的匹配
    # 这里需要实现一个函数来找到全局点云中的点与新图像中的特征点之间的匹配
    # 这通常涉及到搜索全局点云中与新图像特征点最接近的点
    matched_3d_points, matched_2d_points = find_matches(global_point_cloud, keypoints_new, descriptors_new)

    # 使用solvePnP估计新视图的相机姿态
    _, rvec, tvec = cv2.solvePnP(matched_3d_points, matched_2d_points, K, None)

    # 转换旋转向量到旋转矩阵
    R_new, _ = cv2.Rodrigues(rvec)
    T_new = np.hstack((R_new, tvec.reshape(3, 1)))

    # 更新全局相机姿态列表
    global_camera_poses.append(T_new)

    # 可选：使用当前视图更新全局点云
    # 这可能包括三角测量新的点或更新现有点的位置

# Bundle Adjustment (BA) 优化
# 这里可以添加BA的代码来优化相机姿态和3D点云
# 定义BA优化函数
# 使用BA优化相机姿态和三维点云
optimized_camera_poses, optimized_point_cloud = bundle_adjustment(
    global_camera_poses, global_point_cloud, keypoints_all, K, max_iterations=10
)
# 清除错误的3D点
# 这里可以添加代码来检查和清除不合格的3D点

# 添加更多的视图，并重复找到2D-3D对应点和PnP来估计新视图的R和t，以及一致性集C
# 这里可以继续添加更多的视图并重复之前的流程，找到新视图的R和t，以及一致性集C

# 重复上述过程直到添加了所有视图

# 最后，你可以保存全局点云和相机姿态供后续使用
