from style_classifier import *
from unet_ddpm import *
import sys
sys.path.append('./bonibox_gen')
from bonibox_gen.train_box_generator import Box_Lstm
from bonibox_gen.simplebox import generate_box_simple, put_char, visualize_bounding_boxes
from bonibox_gen.generatebox import unconditional_generate_boxes, conditional_generate_boxes


##### utils #####
def draw(traj, i, fake, method='unconditional', line_id = 0):
    seq = traj.copy()  # 不改变原序列
    #seq[:, 0:2] = np.cumsum(traj[:, 0:2], axis=0)
    strokes = np.split(seq, np.where(seq[:, 2] == -1)[0] + 1)  # split[x,y]到x前一个截止
    strokes.pop()
    for s in strokes:
        plt.plot(s[:, 0], -s[:, 1], linewidth=0.5, color='blue')
        #plt.plot(s[:, 0], -s[:, 1], linewidth=0.5)

    ax = plt.gca()
    ax.set_aspect(1)
    #plt.show()
    if fake:
        plt.savefig(f'./paper/{method}{i}-{line_id}.png',dpi=700,bbox_inches="tight")
    else:
        plt.savefig(f'./paper/true{i}-{line_id}.png',dpi=700,bbox_inches="tight")
    plt.close()


def get_ids(tags, duration):
    ids = []
    text = ''
    for i in range(len(tags)):
        char = tags[i]
        text += char
        index = full_dict.index(char)
        item = [index]*int(duration[i])
        item[0] = 4053
        item[-1] = 4054
        ids.append(item)
    return ids, text


def parser_label(label):
    ## 输入一个label list 返回其中每个字符的长度
    duration = []
    l = 0
    total = len(label)

    for k in range(total):
        if label[k] == 4052:
            duration.append(total - k)
            return duration
        elif label[k] == 4054:
            l += 1
            duration.append(l)
            l = 0
        else:
            l += 1

    return duration


def pad(line, k=8):
    # 使line  b l 3 整除k
    b, l, _ = line.shape
    pad_l = k - (l % k)
    tensor = torch.tensor([0,0,-1]).unsqueeze(0).repeat(b, pad_l, 1).to(device)
    line = torch.cat([line, tensor], dim=1)

    return line


def cal_box(traj):
    x_max = np.max(traj[:, 0])
    x_min = np.min(traj[:, 0])
    y_max = np.max(traj[:, 1])
    y_min = np.min(traj[:, 1])

    h = y_max-y_min
    w = x_max-x_min
    center = (y_max + y_min)/2

    return h, w, center, x_min, x_max


def get_style_ref(line_datas, writer_id):
    ref = np.zeros((1, 1000, 3))
    data = line_datas[writer_id]
    line_num = data['line_num']
    l = 0
    while(l<=1000):
        line_id =  np.random.choice(line_num, 1, replace=False)
        l = data[f'{line_id[0]}'].shape[0]
    end = np.random.randint(1000, l)
    #ref[0] = data[f'{line_id[0]}'][end-1000: end]
    ref[0] = data[f'{line_id[0]}'][:1000]

    ref = torch.from_numpy(ref).float().to(device)

    return ref


############  important function   ###########
def get_line(line_datas, writer_id, round): # writer_id: 0 - 1000 - 1218
    data = line_datas[writer_id]
    line_num = data['line_num']

    line_id = round
    tag = ['']
    while('' in tag):
        #line_id  = np.random.randint(0, line_num)
        line_id += 1
        tag = data[f'{line_id}tag']

    tag = data[f'{line_id}tag']
    len_tag = len(tag)
    labels, text = get_ids(tag, data[f'{line_id}duration'])
    leng = data[f'{line_id}'].shape[0]
    line = np.zeros((1, leng, 3))
    line[0, :, :] = data[f'{line_id}']
    line[0, -1, 2] = -1.
    line = torch.from_numpy(line).float().to(device)
    line = pad(line, 8)
    style_refs = get_style_ref(line_datas, writer_id)
    style_refs = style_refs.repeat(len_tag, 1, 1)

    l = max(data[f'{line_id}duration'])
    l += 8 - (l % 8)
    l = max(120, l)
    durations = []
    for label in labels:
        if len(label) < l:
            label += [4052] * (l - len(label))
        else:
            label = label[:l]
        duration = parser_label(label)
        durations.append(duration)

    # 计算文本行边框
    box = np.zeros((1, len_tag, 4))
    box_ref = np.zeros((1, len_tag, 6))

    line_ = data[f'{line_id}'].copy()
    line_[:, 0:2] = np.cumsum(data[f'{line_id}'][:, 0:2], axis=0)  # 绝对坐标
    x_max_ = 0.
    duration = np.array([0] + data[f'{line_id}duration'])
    d = np.cumsum(duration)
    for char_id in range(len_tag):
        char_traj = line_[d[char_id]:d[char_id + 1], :]
        h, w, center, x_min, x_max = cal_box(char_traj)
        box[0, char_id, 0] = h
        box[0, char_id, 1] = w
        box[0, char_id, 2] = center
        box[0, char_id, 3] = x_min - x_max_
        x_max_ = x_max

        char = tag[char_id]
        index = dictionary.index(char)
        box_ref[0, char_id, 0] = char_boxes[index]['h'][0]
        box_ref[0, char_id, 1] = char_boxes[index]['h'][1]
        box_ref[0, char_id, 2] = char_boxes[index]['w'][0]
        box_ref[0, char_id, 3] = char_boxes[index]['w'][1]
        box_ref[0, char_id, 4] = char_boxes[index]['center'][0]
        box_ref[0, char_id, 5] = char_boxes[index]['center'][1]
    box = torch.from_numpy(box).float().to(device)
    box_ref = torch.from_numpy(box_ref).float().to(device)

    return line, text, labels, durations, style_refs, box, box_ref


def generate_chars(model, style_en, labels, durations, style_refs, classifier=None):
    model.eval()
    with torch.no_grad():
        style_refs = style_refs.permute(0, 2, 1)
        x, en1, en2, en3 = style_en(style_refs)
        style_features = [x, en1, en2, en3]

        for r in range(1):
            if classifier == None:
                traj = model.generate(labels, durations, style_features).permute(0, 2, 1) #b,num,_tag,3
            else:
                traj = model.guided_generate(labels, durations, classifier, style_features).permute(0, 2, 1) #b,num,_tag,3

            traj[:, :, 2] = torch.sgn(traj[:, :, 2])
            traj = traj.cpu().numpy()
            traj[:, :, 0:2] = np.cumsum(traj[:, :, 0:2], axis=1)
            #for i in range(min(10, traj.shape[0])):
            #    draw(traj[i], i=i+10*r, fake=True)
            print(traj.shape)
            print('finished')
        
        return traj


def generate_line(writer_id, model, style_en, box_lstm=None, method='unconditional', len_prefix=10, classifier=None):
    for round in range(1):
        line, text, labels, durations, style_refs, box, box_ref = get_line(line_datas, writer_id, round)
        print(text)
        line = line.cpu().numpy()
        line[:, :, 0:2] = np.cumsum(line[:, :, 0:2], axis=1)
        #draw(line[0], writer_id, fake=False, line_id=round)

        chars = generate_chars(model, style_en, labels, durations, style_refs)
        if method == 'conditional':
            boxes = conditional_generate_boxes(box_lstm, box, box_ref, len_prefix, device)
        elif method == 'unconditional':
            boxes = unconditional_generate_boxes(box_lstm, text, dictionary, char_boxes, device)
        elif method == 'simple':
            boxes = generate_box_simple(text, dictionary, char_boxes)
        
        ### 拼接文本行 ###
        line_traj = np.array([[0, 0, -1]])
        num = chars.shape[0]
        for i in range(num):
            traj_ = put_char(chars[i], boxes[i])
            line_traj = np.concatenate((line_traj, traj_), axis=0)
        #draw(line_traj, writer_id, fake=True, method=method, line_id=round)
    
    return 0


def check_generated_boxes(writer_id, box_lstm=None, method='unconditional', len_prefix=10):
    for round in range(1):
        line, text, labels, durations, style_refs, box, box_ref = get_line(line_datas, writer_id, round)
        print(text)
        line = line.cpu().numpy()
        line[:, :, 0:2] = np.cumsum(line[:, :, 0:2], axis=1)
        #draw(line[0], writer_id, fake=False, line_id=round)

        #chars = generate_chars(model, style_en, labels, durations, style_refs)
        if method == 'conditional':
            boxes = conditional_generate_boxes(box_lstm, box, box_ref, len_prefix, device)
        elif method == 'unconditional':
            boxes = unconditional_generate_boxes(box_lstm, text, dictionary, char_boxes, device)
        elif method == 'gaussian':
            boxes = generate_box_simple(text, dictionary, char_boxes)
        #### here boxes is  y_max y_min x_max x_min

        ### 可视化布局 ###
        box = box.cpu().squeeze(0).numpy()
        #print(box)
        gt_boxes = []
        x_right = 0
        for i in range(len(box)):
            height = box[i, 0]
            width  = box[i, 1]
            center = box[i, 2]
            delta  = box[i, 3]

            ymax = center + height/2
            ymin = center - height/2
            xmin = x_right + delta#  后续加偏移
            xmax = xmin + width

            gt_boxes.append([ymax, ymin, xmax, xmin])
            x_right = xmax
        visualize_bounding_boxes(boxes, gt_boxes=gt_boxes, method=method, save_path=f'./paper/boxes/{method}_{writer_id}.png')
    
    return 0



############### main ###############
##### load data #####
print('####### loda datas ########')
char_datas = np.load('/lustre/home/msren/database/char/datas.npy', allow_pickle=True).tolist()  #单字数据训练集
full_dict = np.load('/lustre/home/msren/database/char/mydict.npy', allow_pickle=True).tolist()  #单字字典  4052:EOS  4053:Start  4054:End
full_dict_size = len(full_dict)  #4055

line_datas = np.load('/lustre/home/msren/database/line/all_datas.npy', allow_pickle=True)                   #文本行数据集
dictionary = np.load('/lustre/home/msren/database/line/dictionary.npy', allow_pickle=True).tolist()    #文本行字典
char_boxes = np.load('./bonibox_gen/char_boxes.npy', allow_pickle=True)                                            #保存文本行中每个字在行中的位置信息
print('####### loda datas finished ########')

if __name__ == '__main__':
    ### load model ###
    print('####### create models ########')
    ddpm = DDPM()
    ddpm.load_state_dict(torch.load('./models/unetddpm450000.pth', map_location='cpu'))
    ddpm.to(device)

    style_en = style_encoer()
    style_en.load_state_dict(torch.load('./models/style_encoder450000.pth', map_location='cpu'))
    style_en.to(device)

    box_genertaor = Box_Lstm(device)
    box_genertaor.load_state_dict(torch.load('./bonibox_gen/boxlstm3000.pth', map_location='cpu'))
    box_genertaor.to(device)
    print('####### create models successfully! ########')

    for writer_id in range(1046,1047):
        '''_ = generate_line(writer_id, ddpm, style_en, box_genertaor, method='conditional')
        _ = generate_line(writer_id, ddpm, style_en, box_genertaor, method='unconditional')
        _ = generate_line(writer_id, ddpm, style_en, box_genertaor, method='simple')'''
        #_ = check_generated_boxes(writer_id, box_genertaor, method='conditional')
        #_ = check_generated_boxes(writer_id, box_genertaor, method='unconditional')
        _ = check_generated_boxes(writer_id, box_genertaor, method='gaussian')