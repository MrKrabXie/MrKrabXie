# 读取用户上传的PLY文件，检查其格式和内容是否正确

ply_file_path = 'C:/Users/crab/PycharmProjects/3dReset/results/output1.ply'

def check_ply_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

            # 输出文件的前几行和最后几行以进行检查
            print("File Start:")
            print(''.join(lines[:10]))  # 显示文件开始的10行
            print("File End:")
            print(''.join(lines[-10:]))  # 显示文件结束的10行

            return "Check completed"
    except Exception as e:
        return str(e)

check_ply_file(ply_file_path)
