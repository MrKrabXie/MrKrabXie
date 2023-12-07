# 重新读取PLY文件，专注于修复头部部分

def fix_ply_header(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # 检查并修复头部问题
        new_lines = []
        header_ended = False
        for line in lines:
            # 检查并结束头部
            if "end_header" in line:
                header_ended = True
                new_lines.append("end_header\n")  # 确保头部结束标记正确
                break
            else:
                new_lines.append(line)

        # 如果头部没有正确结束，这是一个问题
        if not header_ended:
            return "Header not properly formatted."

        # 将头部之后的所有行添加到新的列表中
        data_start_index = lines.index("end_header\n") + 1
        new_lines.extend(lines[data_start_index:])

        # 重新写入修正后的文件
        edited_file_path = file_path.replace('.ply', '_header_fixed.ply')
        with open(edited_file_path, 'w') as file:
            file.writelines(new_lines)

        return edited_file_path
    except Exception as e:
        return str(e)


fixed_header_ply_path = fix_ply_header('C:/Users/crab/PycharmProjects/3dReset/results/output1.ply')
fixed_header_ply_path
