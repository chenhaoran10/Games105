import numpy as np
from scipy.spatial.transform import Rotation as R


def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i + 1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1, -1))
        # 把数据拼接成矩阵，行是每一帧的数据
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data


def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """

    joint_name = []
    joint_parent = []
    joint_stack_list = []
    offset_list = []
    joint_dict = {}

    # 'r' ->read only mode
    # {} dictionary ,[] list
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = [name for name in lines[i].split()]
            next_line = [name for name in lines[i + 1].split()]
            if line[0] == "HIERARCHY":
                continue
            if line[0] == "MOTION":
                break
            if line[0] == "ROOT" or line[0] == "JOINT":
                # line[-1] :last element
                joint_name.append(line[-1])
                joint_stack_list.append(line[-1])
            if line[0] == "End":
                joint_name.append(joint_name[-1] + "_end")
                joint_stack_list.append(joint_name[-1])
            if line[0] == "OFFSET":
                offset_list.append([float(line[1]), float(line[2]), float(line[3])])
            if line[0] == "}":
                joint_index = joint_stack_list.pop()
                if not joint_stack_list:
                    continue
                else:
                    # joint_dict[rWrist_end]= rWrist
                    joint_dict[joint_index] = joint_stack_list[-1]
        for i in joint_name:
            if i == "RootJoint":
                joint_parent.append(-1)
            else:
                # parent bone index in joint_name array
                joint_parent_name = joint_dict[i]
                joint_parent.append(joint_name.index(joint_parent_name))
        # expand array form one dimension to two dimension
        joint_offset = np.array(offset_list).reshape(-1, 3)
        print(joint_name)
        # print(joint_offset, type(joint_offset), joint_offset.shape)
        print(joint_parent)
        # exit()
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """

    joint_positions = []
    joint_orientations = []
    end_index = []

    # record end_index in joint_name array
    for i in joint_name:
        if "_end" in i:
            end_index.append(joint_name.index(i))

    frame_data = motion_data[frame_id]
    # 得到了这一帧的数据，并且自动分成了n行3列。
    frame_data = frame_data.reshape(-1, 3)
    # 忽略第一行的三个数，剩下的组成一个矩阵。
    quaternion = R.from_euler('XYZ', frame_data[1:], degrees=True).as_quat()

    for i in end_index:
        # 沿着行序，将骨骼end的数据也插入四元数矩阵中，前面记录了i，所以这里是合适的位置
        # [0, 0, 0, 1]表示没有旋转
        quaternion = np.insert(quaternion, i, [0, 0, 0, 1], axis=0)

    # 同时遍历索引和值
    for index, parent in enumerate(joint_parent):
        if parent == -1:
            # -1说明是root节点，将其数据加入两个LIST
            joint_positions.append(frame_data[0])
            joint_orientations.append(quaternion[0])
        else:
            # 如果有父节点，那么需要考虑父级，进行旋转和平移
            quat = R.from_quat(quaternion)
            # 父级的旋转作用上当前的旋转得到当前的世界旋转，这里需要左乘
            rotation = R.as_quat(quat[index] * quat[parent])
            joint_orientations.append(rotation)
            joint_orientations_quat = R.from_quat(joint_orientations)

            # 计算原来相当于父级的offset应用上这个旋转后的offset，也就是说这个关节旋转后应该在哪
            offset_rotation = joint_orientations_quat[parent].apply(joint_offset[index])
            # 关节的位置 = 父级位置+旋转原来offset后的结果
            joint_positions.append(joint_positions[parent] + offset_rotation)

    joint_positions = np.array(joint_positions)
    joint_orientations = np.array(joint_orientations)
    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    motion_data = None
    return motion_data
