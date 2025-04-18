# -*- coding: utf-8 -*-
from pyexpat import model
from generatebox import *
from simplebox import *



datas = np.load('/lustre/home/msren/database/line/all_datas.npy', allow_pickle=True)              #文本行数据集
dictionary = np.load('/lustre/home/msren/database/line/dictionary.npy', allow_pickle=True).tolist()    #文本行字典
char_boxes = np.load('./char_boxes.npy', allow_pickle=True)                                #保存文本行中每个字在行中的位置信息



def binary_feature(gt_boxes):
    l = len(gt_boxes)
    binary_features = []
    for i in range(1, l):
        f1 = gt_boxes[i][2] - gt_boxes[i-1][2]
        f2 = gt_boxes[i][3] + gt_boxes[i-1][1]/2 + gt_boxes[i][1]/2
        f3 = gt_boxes[i][2] - gt_boxes[i-1][2] + gt_boxes[i][0]/2 - gt_boxes[i-1][0]/2
        f4 = gt_boxes[i][2] - gt_boxes[i-1][2] - gt_boxes[i][0]/2 + gt_boxes[i-1][0]/2
        f5 = gt_boxes[i][3] + gt_boxes[i-1][1]
        f6 = gt_boxes[i][3] + gt_boxes[i][1]
        if gt_boxes[i-1][0] == 0:
            f7 = gt_boxes[i][0] / 1
        else:
            f7 = gt_boxes[i][0] / gt_boxes[i-1][0]
        if gt_boxes[i-1][1] == 0:
            f8 = gt_boxes[i][1] / 1
        else:
            f8 = gt_boxes[i][1] / gt_boxes[i-1][1]

        binary_features.append([f1,f2,f3,f4,f5,f6,f7,f8])

    return binary_features

def generate_box_(text):
    #根据所给文本，生成每个字的框 for eva
    x_left = 0
    boxes = []
    for char in text:
        id = dictionary.index(char)
        height = np.random.normal(char_boxes[id]['h'][0], min(char_boxes[id]['h'][1], 0.06))
        width = np.random.normal(char_boxes[id]['w'][0], min(char_boxes[id]['w'][1], 0.06))
        center = np.random.normal(char_boxes[id]['center'][0], min(char_boxes[id]['center'][1], 0.05))
        delta = np.random.normal(0.015, 0.001)

        boxes.append([height, width, center, delta])

    return boxes


def eval_method(method = 'simple', len_prefix = 15):
    ### load model ###
    #print('####### create lstm models ########')
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = Box_Lstm(device)
    model.load_state_dict(torch.load('/lustre/home/msren/end2end_generate_line/bonibox_gen/boxlstm3000.pth', map_location='cpu'))
    model.to(device)
    #print('####### create lstm models successfully! ########')

    dist = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0.])
    num = 0

    for writer in range(1000, 1050):

        data = datas[writer]
        line_num = datas[writer]['line_num']
        for line_id in range(line_num):
            # 由于长度不同 只能一行一行的算
            # 计算GT的文本框信息
            line = data[f'{line_id}'].copy()
            line[:, 0:2] = np.cumsum(data[f'{line_id}'][:, 0:2], axis=0)  # 绝对坐标
            # 文本行标签
            tag = data[f'{line_id}tag']
            len_tag = len(tag)

            x_max_ = 0.
            duration = np.array([0] + data[f'{line_id}duration'])
            d = np.cumsum(duration)
            gt_boxes = []
            for char_id in range(len_tag):
                char_traj = line[d[char_id]:d[char_id + 1], :]
                h, w, center, x_min, x_max = cal_box(char_traj)
                delta = x_min - x_max_
                x_max_ = x_max
                char_box = [h, w, center, delta]
                gt_boxes.append(char_box)
            
            binary_features = torch.tensor(binary_feature(gt_boxes))
            gt_boxes = torch.tensor(gt_boxes)

            ###### 对于采样的方法 ######
            if method == 'simple':
                boxes = generate_box_(tag)
                bfs = torch.tensor(binary_feature(boxes))
                boxes = torch.tensor(boxes)
                num += 1
                #dist += torch.mean(torch.abs(gt_boxes-boxes), 0)
                dist += torch.mean(torch.abs(binary_features-bfs), 0)

            elif method == 'conditional' and len_tag > (len_prefix+1):
                batch = np.zeros((1, len_tag, 4))
                batch_ref = np.zeros((1, len_tag, 6))
                for char_id in range(len_tag):
                    char_traj = line[d[char_id]:d[char_id + 1], :]
                    h, w, center, x_min, x_max = cal_box(char_traj)
                    batch[0, char_id, 0] = h
                    batch[0, char_id, 1] = w
                    batch[0, char_id, 2] = center
                    batch[0, char_id, 3] = x_min - x_max_
                    x_max_ = x_max

                    char = tag[char_id]
                    index = dictionary.index(char)
                    batch_ref[0, char_id, 0] = char_boxes[index]['h'][0]
                    batch_ref[0, char_id, 1] = char_boxes[index]['h'][1]
                    batch_ref[0, char_id, 2] = char_boxes[index]['w'][0]
                    batch_ref[0, char_id, 3] = char_boxes[index]['w'][1]
                    batch_ref[0, char_id, 4] = char_boxes[index]['center'][0]
                    batch_ref[0, char_id, 5] = char_boxes[index]['center'][1]

                batch = torch.from_numpy(batch).float().to(device)
                batch_ref = torch.from_numpy(batch_ref).float().to(device)

                ######  ######
                target_seq_shift = torch.zeros_like(batch)
                target_seq_shift[:, 1:, :] = batch[:, :-1, :]

                prefix_seq = model.func_in(batch_ref[:, :len_prefix, :])
                prefix_seq = torch.cat([target_seq_shift[:, :len_prefix, :], prefix_seq], -1)
                ref_seq = batch_ref[:, len_prefix:, :]
                box = model.get_box(ref_seq, prefix_seq)
                box = torch.cat(box).detach().cpu() ####box的参数  l bs=1 4
                box = box[:, 0, :]
                num += 1
                bfs = torch.tensor(binary_feature(box))
                #dist += torch.mean(torch.abs(gt_boxes[len_prefix: , :] - box), 0)
                #print(bfs.shape, binary_features.shape)
                dist += torch.mean(torch.abs(binary_features[len_prefix:, :] - bfs), 0)
            
            elif method == 'unconditional':
                batch_ref = np.zeros((1, len_tag, 6))
                for char_id in range(len_tag):
                    char = tag[char_id]
                    index = dictionary.index(char)
                    batch_ref[0, char_id, 0] = char_boxes[index]['h'][0]
                    batch_ref[0, char_id, 1] = char_boxes[index]['h'][1]
                    batch_ref[0, char_id, 2] = char_boxes[index]['w'][0]
                    batch_ref[0, char_id, 3] = char_boxes[index]['w'][1]
                    batch_ref[0, char_id, 4] = char_boxes[index]['center'][0]
                    batch_ref[0, char_id, 5] = char_boxes[index]['center'][1]
                
                batch_ref = torch.from_numpy(batch_ref).float().to(device)
                box = model.get_box(batch_ref)
                box = torch.cat(box).detach().cpu() ####box的参数  l bs=1 4
                box = box[:, 0, :]
                num += 1
                bfs = torch.tensor(binary_feature(box))
                #dist += torch.mean(torch.abs(gt_boxes-box), 0)
                dist += torch.mean(torch.abs(binary_features-bfs), 0)

    dist /= num
    print(len_prefix, dist)


if __name__ == '__main__':
    ### simple conditional unconditional ###
    for i in range(15, 20):
        eval_method(method='conditional', len_prefix=i)