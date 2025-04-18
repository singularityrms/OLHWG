# -*- coding: utf-8 -*-
import imp
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class HParams():
    def __init__(self):
        # Encoder
        # Decoder
        # 训练超参数
        self.bs = 8
        self.epo = 20001
        self.lr = 0.001
        self.min_lr = 0.000001
        self.lr_decay = 0.9995
        self.grad_clip = 1.

        # DDPM
        self.hidden_size = 128
        self.n_layers = 2
        self.dropout = 0.1
        self.emb_size = 32
        self.time_step = 200
        self.beta_0 = 0.002
        self.beta_T = 0.1


hp = HParams()


#### network #######
class Box_Lstm(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.lstm = nn.LSTM(hp.emb_size + 4, hp.hidden_size, hp.n_layers, batch_first=True, dropout=hp.dropout, bidirectional=False)
        self.func_out = nn.Linear(hp.hidden_size, 4)
        self.func_in = nn.Linear(6, hp.emb_size)
        #self.loss = nn.MSELoss()
        self.loss = nn.L1Loss()

        self.device = device

    def forward(self, inputs):
        # x : b L dim (dim 由上一时间步的输出与当前时间步的输入拼接而成)
        hidden = torch.zeros(hp.n_layers, inputs.shape[0], hp.hidden_size).to(self.device)
        cell = torch.zeros(hp.n_layers, inputs.shape[0], hp.hidden_size).to(self.device)
        hidden_cell = (hidden, cell)

        x, _ = self.lstm(inputs, hidden_cell)
        x = self.func_out(x)
        return x

    def get_loss(self, target_seq, ref_seq):
        # during train, input_seq is [zeros, shift(target_seq)]
        # ref seq : b L 6
        target_seq_shift = torch.zeros_like(target_seq)
        target_seq_shift[:, 1:, :] = target_seq[:, :-1, :]
        input_seq = self.func_in(ref_seq)
        input_seq = torch.cat([target_seq_shift, input_seq], -1)

        out_seq = self.forward(input_seq)
        loss = self.loss(target_seq, out_seq)

        return loss
    
    def get_box(self, ref_seq, prefix_seq=None):
        # prefix_seq : b L 4+dim
        # ref_seq : b NUM 6
        bs, num, _ = ref_seq.shape

        if prefix_seq == None:        
            sos = torch.zeros(bs, 1, 4).to(self.device)
            hidden = torch.zeros(hp.n_layers, bs, hp.hidden_size).to(self.device)
            cell = torch.zeros(hp.n_layers, bs, hp.hidden_size).to(self.device)
            hidden_cell = (hidden, cell) 
        else:
            hidden = torch.zeros(hp.n_layers, bs, hp.hidden_size).to(self.device)
            cell = torch.zeros(hp.n_layers, bs, hp.hidden_size).to(self.device)
            hidden_cell = (hidden, cell)

            x, hidden_cell = self.lstm(prefix_seq, hidden_cell)
            x = self.func_out(x)
            sos = x[:,-1,:].unsqueeze(1)

        boxes = []
        for i in range(num):
            inputs = self.func_in(ref_seq[:, i, :]).unsqueeze(1)  # b 1 dim
            inputs = torch.cat([sos, inputs], -1)
            outputs, hidden_cell = self.lstm(inputs, hidden_cell)
            sos = self.func_out(outputs)

            boxes.append(sos)
        
        return boxes


#### train process ######
def lr_decay(optimizer):
    # Decay learning rate by a factor of lr_decay
    for param_group in optimizer.param_groups:
        if param_group['lr'] > hp.min_lr:
            param_group['lr'] *= hp.lr_decay
    return optimizer


def cal_box(traj):
    x_max = np.max(traj[:, 0])
    x_min = np.min(traj[:, 0])
    y_max = np.max(traj[:, 1])
    y_min = np.min(traj[:, 1])

    h = y_max-y_min
    w = x_max-x_min
    center = (y_max + y_min)/2

    return h, w, center, x_min, x_max


def get_batch(bs=32, train=True):
    char_num_max = 25
    batch = np.zeros((bs, char_num_max, 4))
    batch_ref = np.zeros((bs, char_num_max, 6))

    if train:
        writers = np.random.randint(0, 1000, bs)
    else:
        writers = np.random.choice(0, 219) + 1000
    
    for k in range(bs):
        writer = writers[k]
        data = datas[writer]
        line_num = datas[writer]['line_num']
        line_id  = np.random.randint(0, line_num)

        line = data[f'{line_id}'].copy()
        line[:, 0:2] = np.cumsum(data[f'{line_id}'][:, 0:2], axis=0)  # 绝对坐标
        # 文本行标签
        tag = data[f'{line_id}tag']
        len_tag = len(tag)

        x_max_ = 0.
        duration = np.array([0] + data[f'{line_id}duration'])
        d = np.cumsum(duration)

        for char_id in range(min(len_tag, char_num_max)):
            char_traj = line[d[char_id]:d[char_id + 1], :]
            h, w, center, x_min, x_max = cal_box(char_traj)
            batch[k, char_id, 0] = h
            batch[k, char_id, 1] = w
            batch[k, char_id, 2] = center
            batch[k, char_id, 3] = x_min - x_max_
            x_max_ = x_max

            char = tag[char_id]
            index = dictionary.index(char)
            batch_ref[k, char_id, 0] = char_boxes[index]['h'][0]
            batch_ref[k, char_id, 1] = char_boxes[index]['h'][1]
            batch_ref[k, char_id, 2] = char_boxes[index]['w'][0]
            batch_ref[k, char_id, 3] = char_boxes[index]['w'][1]
            batch_ref[k, char_id, 4] = char_boxes[index]['center'][0]
            batch_ref[k, char_id, 5] = char_boxes[index]['center'][1]

    batch = torch.from_numpy(batch).float().to(device)
    batch_ref = torch.from_numpy(batch_ref).float().to(device)

    return batch, batch_ref


def train_box_generator(model, epochs=hp.epo):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)

    for epoch in range(epochs):
        target_seq, ref_seq = get_batch()
        loss = model.get_loss(target_seq, ref_seq)

        optimizer.zero_grad()
        print(f'#### batch : {epoch}; loss : {loss.item()}')
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
        optimizer.step()
        optimizer = lr_decay(optimizer)

        # save model
        if epoch>1000 and epoch%3000 == 0:
            print('------saving model-----')
            torch.save(model.state_dict(), './boxlstm{}.pth'.format(epoch))


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('####### loda datas ########')
    datas = np.load('/lustre/home/msren/database/line/all_datas.npy', allow_pickle=True)
    dictionary = np.load('/lustre/home/msren/database/line/dictionary.npy', allow_pickle=True).tolist()
    char_boxes = np.load('./char_boxes.npy', allow_pickle=True)
    print('####### loda datas finished ########')\
    
    ### load model ###
    print('####### create lstm models ########')
    box_genertaor = Box_Lstm(device)
    box_genertaor.to(device)
    print('####### create lstm models successfully! ########')

    box_genertaor = train_box_generator(box_genertaor)