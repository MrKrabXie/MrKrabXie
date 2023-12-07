import open3d as o3d

def load_and_visualize_ply(ply_file_path):
    # 加载PLY文件
    point_cloud = o3d.io.read_point_cloud(ply_file_path)

    # 可视化
    o3d.visualization.draw_geometries([point_cloud])

# 文件路径
ply_file_path = 'C:/Users/crab/PycharmProjects/3dReset/results/result.ply'
# ply_file_path = 'E:/code/python/smp/results/output1.ply'

# 调用函数
load_and_visualize_ply(ply_file_path)



def display_point_cloud(points, colors):
    # 将点云和颜色转换为open3d可用的格式
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(points)
    point_cloud_o3d.colors = o3d.utility.Vector3dVector(colors / 255)  # 归一化颜色值

    # 显示点云
    o3d.visualization.draw_geometries([point_cloud_o3d])
