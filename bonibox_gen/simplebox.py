# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
####  基于采样 简单生成字符框

def generate_box_simple(text, dictionary, char_boxes):
    #根据所给文本，生成每个字的框

    x_left = 0
    boxes = []
    for char in text:
        id = dictionary.index(char)
        height = np.random.normal(char_boxes[id]['h'][0], min(char_boxes[id]['h'][1], 0.06))
        width = np.random.normal(char_boxes[id]['w'][0], min(char_boxes[id]['w'][1], 0.06))
        center = np.random.normal(char_boxes[id]['center'][0], min(char_boxes[id]['center'][1], 0.05))

        #print(height)
        ymax = center + height/2
        ymin = center - height/2

        xmin = x_left   #  后续加偏移
        xmax = xmin + width

        boxes.append([ymax, ymin, xmax, xmin])
        x_left = xmax + 0.015

    return boxes


def visualize_bounding_boxes(bounding_boxes, gt_boxes, method, save_path='./test.png'):
    #print(method)
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.set_aspect('equal')  # Maintain the aspect ratio

    for bbox in bounding_boxes:
        ymax, ymin, xmax, xmin = bbox
        # Calculate width and height
        width = xmax - xmin
        height = ymax - ymin
        # Create a rectangle patch
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    
    if gt_boxes is not None:
        for bbox in gt_boxes:
            ymax, ymin, xmax, xmin = bbox
            # Calculate width and height
            width = xmax - xmin
            height = ymax - ymin
            # Create a rectangle patch
            rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='g', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)

    # Set specific limits for x and y axis
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 1.1)
    #ax.set_xlabel('X Coordinate')
    #ax.set_ylabel('Y Coordinate')
    ax.set_title(method.capitalize())
    plt.gca().invert_yaxis()  # Invert y-axis to match typical image coordinates
    plt.grid(False)  # Enable grid
    plt.savefig(save_path, dpi=300)  # Save the figure
    plt.close()


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


def put_char(traj, params):
    # 把一个字符填到准备好的文本框里  traj要求绝对坐标格式
    ymax = params[0]
    ymin = params[1]
    xmax = params[2]
    xmin = params[3]

    traj[:, 0] -= np.min(traj[:, 0])
    traj[:, 1] -= np.min(traj[:, 1])
    traj[:, 0] *= (xmax-xmin) / (np.max(traj[:, 0])-np.min(traj[:, 0]))
    traj[:, 1] *= (ymax - ymin) / (np.max(traj[:, 1]) - np.min(traj[:, 1]))
    traj[:, 0] += xmin
    traj[:, 1] += ymin

    return traj


if __name__ == '__main__':
    dictionary = np.load('/lustre/home/msren/database/line/dictionary.npy', allow_pickle=True).tolist()
    char_boxes = np.load('./char_boxes.npy', allow_pickle=True)
    '''
    char_boxes:
    list 3054
    char_boxes[id] : dictionary
        ['h'] ['w'] ['center']
    '''

    def draw(traj, i):
        seq = traj.copy()  # 不改变原序列
        strokes = np.split(seq, np.where(seq[:, 2] == -1)[0] + 1)  # split[x,y]到x前一个截止
        strokes.pop()
        for s in strokes:
            plt.plot(s[:, 0], -s[:, 1], color='black', linewidth=1.)

        ax = plt.gca()
        ax.set_aspect(1)
        plt.show()
        plt.savefig('./img/test{}.png'.format(i), dpi=700, bbox_inches="tight")
        plt.close()

    char_datas = np.load('/lustre/home/msren/database/line/char_datas.npy', allow_pickle=True).tolist()
    text = '人人人人人人人人'
    box = generate_box(text)
    line_traj = np.array([[0, 0, 0]])
    for i in range(len(text)):
        c = text[i]
        id = dictionary.index(c)
        char_data = char_datas[id]
        #随便选一个人
        num = len(list(char_data.items()))
        writer = np.random.randint(0, num)
        print(f'{c} : writer: {list(char_data.items())[writer][0]}')
        traj = list(char_data.items())[writer][1]
        traj_ = traj.copy()
        traj_[:, 0:2] = np.cumsum(traj[:, 0:2], axis=0)
        traj_ = put_char(traj_, box[i])
        line_traj = np.concatenate((line_traj, traj_), axis=0)

    draw(line_traj[1:])

