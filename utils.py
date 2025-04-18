import numpy as np
from fastdtw import fastdtw
import math

def clear_traj(traj):
    # 去掉最后的padding
    l = len(traj)
    for i in range(1,l):
        if traj[l-i,2] == 1:
            j = i
            break
    
    traj = traj[:l-j+1]
    return traj


## dowsamoling
def rdp(points, epsilon):
    if len(points) < 3:
        return points

    dmax = 0
    index = 0

    start_point = points[0]
    end_point = points[-1]

    # 计算所有点到起始点和结束点之间的垂直距离
    for i in range(1, len(points) - 1):
        d = perpendicular_distance(points[i], start_point, end_point)
        if d > dmax:
            dmax = d
            index = i
    result = []

    # 如果最大距离大于阈值epsilon，则递归地应用RDP算法
    if dmax > epsilon:
        recursive_results1 = rdp(points[:index+1], epsilon)
        recursive_results2 = rdp(points[index:], epsilon)

        # 合并两个子结果
        result = recursive_results1[:-1] + recursive_results2
    else:
        result = [start_point, end_point]

    return result


def perpendicular_distance(point, start_point, end_point):
    x, y, p = point
    x1, y1, p1 = start_point
    x2, y2, p2 = end_point

    # 计算点到线段的垂直距离
    if p == -1:
        distance = float('inf')
    elif x2 == x1:
        distance = abs(x - x1)
    else:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        distance = abs(slope * x - y + intercept) / np.sqrt(slope**2 + 1)

    return distance


def down_sample(traj, epsilon=0.02):
    seq = traj.copy()
    #seq[:, 0:2] = np.cumsum(traj[:, 0:2], axis=0)
    seq = seq.tolist()          
    #  使用RDP算法简化数据
    simplified_seq = rdp(seq, epsilon)
    simplified_seq = np.array(simplified_seq)
    '''delta_x = simplified_seq[1:, 0] - simplified_seq[0:-1, 0]
    delta_y = simplified_seq[1:, 1] - simplified_seq[0:-1, 1]
    simplified_seq[1:, 0] = delta_x
    simplified_seq[1:, 1] = delta_y'''

    traj = simplified_seq
    return traj


## dtw 
def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def total_euclidean_distance(points):
    n = len(points)
    total_distance = 0.0
    
    for i in range(n-1):
            total_distance += euclidean_distance(points[i], points[i+1])
    
    return total_distance


def center_and_normalize(traj):
    sequence = traj[:,:2]
    # 计算重心
    centroid = np.mean(sequence, axis=0)
    # 平移使重心位于原点
    centered_sequence = sequence - centroid
    # 计算高度（y坐标的范围）
    height = np.max(centered_sequence[:, 1]) - np.min(centered_sequence[:, 1])
    # 保持长宽比，将高度缩放到1
    normalized_sequence = centered_sequence / height
    traj[:,:2] = normalized_sequence
    return traj


def norm_dtw(seq_1, seq_2):
    fast_d, _ = fastdtw(seq_1, seq_2)
    len1 = total_euclidean_distance(seq_1)
    len2 = total_euclidean_distance(seq_2)
    norm_d = fast_d/len2

    return norm_d


if __name__ == '__main__':
    char_datas = np.load('/lustre/home/msren/database/char/test_datas.npy', allow_pickle=True)
    mydict = np.load('/lustre/home/msren/database/char/mydict.npy', allow_pickle=True).tolist()

    ds = []
    for id in range(200):
        char0 = char_datas[0][id][0]
        traj0 = char_datas[0][id][1]
        traj0[:, 0:2] = np.cumsum(traj0[:, 0:2], axis=0)
        traj0 = down_sample(traj0)
        traj0 = center_and_normalize(traj0)
        #print(len(traj0))

        num_writer = len(char_datas)
        lens = []
        for writer in range(1, num_writer):
            datas = char_datas[writer]
            for i in range(len(datas)):
                if datas[i][0] == char0:
                    traj1 = datas[i][1]
                    traj1[:, 0:2] = np.cumsum(traj1[:, 0:2], axis=0)
                    traj1 = down_sample(traj1)
                    traj1 = center_and_normalize(traj1)

                    norm_d = norm_dtw(traj0[:, :2], traj1[:, :2])
                    ds.append(norm_d)
                    lens.append(len(traj1))
    
    print(np.mean(ds))
