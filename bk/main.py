import cv2
import numpy as np
import os

# 文件路径请替换成您自己的路径
img_dir = "C:/Users/crab/PycharmProjects/3dReset/castle-P19/images/"

ply_save_path = "C:/Users/crab/PycharmProjects/3dReset/results/output1.ply"

# 相机内参矩阵K
K = np.array([[2759.48, 0, 1520.69],
              [0, 2764.16, 1006.81],
              [0, 0, 1]])

# 初始化SIFT特征提取器
sift = cv2.SIFT_create()

# 读取图像并提取特征点和描述子
def load_and_find_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f"无法读取图像: {image_path}")
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return image, keypoints, descriptors

# 使用FLANN进行特征匹配
def flann_feature_matches(descriptors1, descriptors2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    return good_matches

# 估计基础矩阵并使用RANSAC筛选内点
def find_fundamental_and_filter(matches, keypoints1, keypoints2):
    matchedPoints1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    matchedPoints2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    F, inliers = cv2.findFundamentalMat(matchedPoints1, matchedPoints2, cv2.FM_RANSAC)
    inliers = inliers.ravel().astype(bool)
    return matchedPoints1[inliers], matchedPoints2[inliers], F

# 从匹配的特征点重建3D点
def reconstruct_3d_points(K, matchedPoints1, matchedPoints2, F):
    E = K.T @ F @ K
    _, R, t, _ = cv2.recoverPose(E, matchedPoints1, matchedPoints2, K)
    proj1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
    proj2 = np.hstack((R, t))
    proj1 = K @ proj1
    proj2 = K @ proj2
    points4D_homogeneous = cv2.triangulatePoints(proj1, proj2, matchedPoints1.T, matchedPoints2.T)
    point3D = cv2.convertPointsFromHomogeneous(points4D_homogeneous.T)
    return point3D.reshape(-1, 3)

# 保存点云为PLY文件
def save_ply_file(points, colors, filename):
    with open(filename, 'w') as plyFile:
        plyFile.write("ply\n")
        plyFile.write("format ascii 1.0\n")
        plyFile.write("element vertex {}\n".format(len(points)))
        plyFile.write("property float x\n")
        plyFile.write("property float y\n")
        plyFile.write("property float z\n")
        plyFile.write("property uchar blue\n")
        plyFile.write("property uchar green\n")
        plyFile.write("property uchar red\n")
        plyFile.write("end_header\n")
        for point, color in zip(points, colors):
            plyFile.write("{} {} {} {} {} {}\n".format(point[0], point[1], point[2], color[2], color[1], color[0]))

# 获取颜色信息的函数
def get_color(image, pt):
    x, y = int(pt[0]), int(pt[1])
    if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
        return [0, 0, 0]  # 如果坐标越界，返回黑色
    return image[y, x]

try:
    # 以第一幅图像为基础，逐步添加后续图像并构建点云
    base_image_path = os.path.join(img_dir, '0000.jpg')
    base_image, base_keypoints, base_descriptors = load_and_find_features(base_image_path)

    # 初始化点云和颜色
    point_cloud = np.empty((0, 3))
    point_colors = np.empty((0, 3))

    # 遍历从'0001.jpg'到'0018.jpg'的图像
    for i in range(1, 19):  # 修改这里的范围以包含所有的图像
    # for i in range(1, 2):  # 修改这里的范围以包含所有的图像
        next_image_path = os.path.join(img_dir, f'{i:04d}.jpg')
        print(f"Processing image: {next_image_path}")
        next_image, next_keypoints, next_descriptors = load_and_find_features(next_image_path)

        # 匹配特征点
        matches = flann_feature_matches(base_descriptors, next_descriptors)

        # 如果没有足够的匹配项，就继续到下一张图
        if len(matches) < 4:
            print(f"Not enough matches found between {base_image_path} and {next_image_path}")
            continue

        # 使用RANSAC筛选内点，并估计基础矩阵
        matchedPoints1, matchedPoints2, F = find_fundamental_and_filter(matches, base_keypoints, next_keypoints)

        # 如果没有足够的内点，就继续到下一张图
        if matchedPoints1.shape[0] < 4:
            print(f"Not enough inliers found between {base_image_path} and {next_image_path}")
            continue

        # 重建3D点
        reconstructed_points = reconstruct_3d_points(K, matchedPoints1, matchedPoints2, F)

        # 添加到点云
        point_cloud = np.vstack((point_cloud, reconstructed_points))

        # # 获取颜色信息
        # colors = [base_image[int(pt[1]), int(pt[0])] for pt in matchedPoints1]
        # point_colors = np.vstack((point_colors, colors))
        # 获取颜色信息
        colors = [get_color(base_image, pt) for pt in matchedPoints1]
        point_colors = np.vstack((point_colors, colors))

        # 更新基准图像和描述子以便下一次迭代
        base_image, base_keypoints, base_descriptors = next_image, next_keypoints, next_descriptors

    # 保存点云到PLY文件
    save_ply_file(point_cloud, point_colors, ply_save_path)
    print(f"PLY file saved to {ply_save_path}")

except Exception as e:
    print(e)
