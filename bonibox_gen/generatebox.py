# -*- coding: utf-8 -*-
from train_box_generator import *
import pdb

def unconditional_generate_boxes():
    return 0

def conditional_generate_boxes(box_lstm, box, box_ref, len_prefix, device):
    ######  ######
    target_seq_shift = torch.zeros_like(box)
    target_seq_shift[:, 1:, :] = box[:, :-1, :]

    prefix_seq = box_lstm.func_in(box_ref[:, :len_prefix, :])
    prefix_seq = torch.cat([target_seq_shift[:, :len_prefix, :], prefix_seq], -1)
    ref_seq = box_ref[:, len_prefix:, :]
    box = box_lstm.get_box(ref_seq, prefix_seq) # list
    box = torch.cat(box).detach().cpu() ####box的参数  l bs=1 4
    box = box[:, 0, :]
    ## 和前缀拼接获得完整的box ##
    target_seq_shift = target_seq_shift.detach().cpu()
    prefix_box = target_seq_shift[0,1:1+len_prefix,:]
    box = torch.cat([prefix_box, box],0)  #完整文本行的box  h w c d
    
    ### 转换成 [ymax ymin xmax xmin]
    bounding_boxes = []
    x_left = 0
    for i in range(box.shape[0]):
        height = box[i,0]
        width = box[i,1]
        center = box[i,2]
        delta = box[i,3]

        ymax = center + height/2
        ymin = center - height/2

        xmin = x_left + delta   #  后续加偏移
        xmax = xmin + width

        bounding_boxes.append([ymax, ymin, xmax, xmin])
        x_left = xmax

    return bounding_boxes
    