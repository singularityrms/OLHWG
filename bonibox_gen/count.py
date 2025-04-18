# -*- coding: utf-8 -*-
import numpy as np


def cal_box(traj):
    x_max = np.max(traj[:, 0])
    x_min = np.min(traj[:, 0])
    y_max = np.max(traj[:, 1])
    y_min = np.min(traj[:, 1])

    h = y_max-y_min
    w = x_max-x_min
    center = (y_max + y_min)/2

    return h, w, center


def gaussian_parameters(arr):
    arr = np.array(arr)
    mu = np.mean(arr)  # 计算均值
    sigma = np.std(arr)  # 计算标准差（方差的平方根）
    return mu, sigma


if __name__ == '__main__':
    print('########  loading data ..... #######')
    datas = np.load('/lustre/home/msren/database/line/all_datas.npy', allow_pickle=True)
    dictionary = np.load('/lustre/home/msren/database/line/dictionary.npy', allow_pickle=True).tolist()
    print('#########  load datas successfully! ########')
    lmax = 2000  # 最长的文本行为2247
    dict_size = len(dictionary)  # 统计为3055

    #### 将每个字的所有数据单独存到一个字典里
    char_boxes = []
    for i in range(dict_size - 3):  # eos start end
        char_box = {}
        h = []
        w = []
        center = []
        char_box.update({'h':h})
        char_box.update({'w': w})
        char_box.update({'center': center})
        char_boxes.append(char_box)

    #### 遍历所有数据 填充
    for writer in range(len(datas)):
        print(f'writer:{writer}')
        data = datas[writer]
        line_num = data['line_num']
        for i in range(line_num):
            line = data[f'{i}'].copy()
            line[:, 0:2] = np.cumsum(data[f'{i}'][:, 0:2], axis=0)  # 绝对坐标
            # 文本行标签
            tag = data[f'{i}tag']
            len_tag = len(tag)

            duration = np.array([0] + data[f'{i}duration'])
            d = np.cumsum(duration)
            for char_id in range(len_tag):
                char_traj = line[d[char_id]:d[char_id + 1], :]
                char = tag[char_id]
                index = dictionary.index(char)

                h, w, center = cal_box(char_traj)
                char_boxes[index]['h'].append(h)
                char_boxes[index]['w'].append(w)
                char_boxes[index]['center'].append(center)


    for i in range(dict_size - 3):
        char_box = char_boxes[i]
        ## height
        h = char_box['h']
        mu, sigma = gaussian_parameters(h)
        char_boxes[i]['h'] = [mu, sigma]

        ## width
        w = char_box['w']
        mu, sigma = gaussian_parameters(w)
        char_boxes[i]['w'] = [mu, sigma]

        ## center
        center = char_box['center']
        mu, sigma = gaussian_parameters(center)
        char_boxes[i]['center'] = [mu, sigma]
        print(char_boxes[i]['h'])


    np.save('char_boxes.npy', char_boxes)

