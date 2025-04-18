from os import device_encoding
from style_classifier import *
from unet_ddpm import *
from utils import *
from train_ddpm import pad, parser_label
from evaluate_char.char_classifier import char_classifier_testset
from evaluate_char.style_classifier import style_encoder_testset

char_datas = np.load('/lustre/home/msren/database/char/test_datas.npy', allow_pickle=True)
mydict = np.load('/lustre/home/msren/database/char/mydict.npy', allow_pickle=True).tolist()

def draw(traj, i=0, fake=True):
    seq = traj.copy()  # 不改变原序列
    #seq[:, 0:2] = np.cumsum(traj[:, 0:2], axis=0)
    strokes = np.split(seq, np.where(seq[:, 2] == -1)[0] + 1)  # split[x,y]到x前一个截止
    strokes.pop()
    for s in strokes:
        plt.plot(s[:, 0], -s[:, 1], linewidth=1.5)

    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()
    if fake:
        plt.savefig('./evaluate_char/img/test{}.png'.format(i),dpi=700,bbox_inches="tight")
    else:
        plt.savefig('./evaluate_char/img/true{}.png'.format(i),dpi=700,bbox_inches="tight")
    plt.close()

    
def get_batch(bs, round):
    labels = []
    lens = []
    lines = np.zeros((bs, 120, 3))
    #ids = np.random.choice(60, bs, replace=True).tolist()
    ids = [round] * bs
    char_ids = []

    for i in range(bs):
        id = ids[i]
        char_data = char_datas[id]
        num = len(char_data)
        j = np.random.randint(0, num)
        char = char_data[j][0]
        char_id = mydict.index(char)
        char_ids.append(char_id)
        traj = char_data[j][1]

        length = traj.shape[0]
        lens.append(length)
        lines[i, :length, :] = traj
        #lines[i, 0, :] = np.array([0., 0., 1.])
        lines[i, length:, 2] = -1.

        label = [char_id] * length  
        label[0] = 4053
        label[-1] = 4054
        labels.append(label)
    
    #max_len = max(lens)
    #lines = lines[:, :max_len, :]
    lines = torch.from_numpy(lines).float().to(device)
    batch = pad(lines, 8)
    style_refs = get_style_batch_10(char_datas, ids)

    l = batch.shape[1]#也等于label的长度
    durations = []
    for label in labels:
        if len(label) < l:
            label += [4052] * (l - len(label))
        else:
            label = label[:l]

        duration = parser_label(label)
        durations.append(duration)
    
    style_ids = torch.tensor(ids).long().to(device)
    char_ids = torch.tensor(char_ids).long().to(device)
    #print(durations)
    return batch, labels, durations, style_refs, char_ids, style_ids


def evaluate_socre(ddpm, style_en, classifier, style_classifier, classifier_=None):
    char_accuray = []
    style_accuray = []
    for round in range(60):
        with torch.no_grad():
            bs = 150
            num = 15
            bs_ = int(bs/num)

            batch, labels, durations, style_refs, char_ids, style_ids = get_batch(bs, round)
            style_refs = style_refs.permute(0, 2, 1)
            x, en1, en2, en3 = style_en(style_refs)
            style_features = [x, en1, en2, en3]
            char_correct = torch.zeros(bs, dtype=torch.bool).to(device)
            style_correct = torch.zeros(int(bs_), dtype=torch.bool).to(device)
            ### GT ###
            batch = batch.detach().cpu().numpy()
            '''for i in range(10):
                draw(batch[i], i=i, fake=False)'''

            ### 生成 ####
            for r in range(1):
                if classifier_ == None:
                    traj = ddpm.generate(labels, durations, style_features).permute(0, 2, 1)
                else:
                    traj = ddpm.guided_generate(labels, durations, classifier_, style_features).permute(0, 2, 1)

                traj[:, :, 2] = torch.sgn(traj[:, :, 2])
                ### 可视化 ###
                '''traj_ = traj.cpu().numpy()
                for i in range(10):
                    draw(traj_[i], i=i+10*r, fake=True)'''

                ### 准确率 ###
                char_correct_ = classifier.acc_function(traj, char_ids)
                style_correct_ = style_classifier.acc_function_(traj, style_ids, num)
                #print(float(torch.sum(char_correct_)) / len(char_correct_), float(torch.sum(style_correct_)) / len(style_correct_))
                char_correct = torch.logical_or(char_correct, char_correct_)
                style_correct = torch.logical_or(style_correct, style_correct_)

            char_correct = torch.sum(char_correct)
            style_correct = torch.sum(style_correct)
            char_acc = float(char_correct) / bs
            style_acc = float(style_correct) / int(bs_)

            #print(f'{round+1} roudn in total 60, char_acc: {char_acc}, style_acc : {style_acc}')
            print(f'{round+1} roudn in total 60, style_acc : {style_acc}')
            char_accuray.append(char_acc) 
            style_accuray.append(style_acc)
            print(style_accuray)
            #print(f'### current acc : {np.mean(char_accuray)} ,  {np.mean(style_accuray)}###')
            print(f'### current style acc : {np.mean(style_accuray)}###')


def evaluate_dtw(ddpm, style_en, classifier, style_classifier, classifier_=None):
    norm_ds = []
    for round in range(60):
        with torch.no_grad():
            bs = 50
            batch, labels, durations, style_refs, char_ids, style_ids = get_batch(bs, round)
            style_refs = style_refs.permute(0, 2, 1)
            x, en1, en2, en3 = style_en(style_refs)
            style_features = [x, en1, en2, en3]

            ### GT ###
            batch = batch.detach().cpu().numpy()
            for r in range(1):
                ### 生成 ####
                if classifier_ == None:
                    traj = ddpm.generate(labels, durations, style_features).permute(0, 2, 1)
                else:
                    traj = ddpm.guided_generate(labels, durations, classifier_, style_features).permute(0, 2, 1)
                traj[:, :, 2] = torch.sgn(traj[:, :, 2])
                traj = traj.cpu().numpy()

                ### dtw ###
                norm_d = 0
                for i in range(bs):
                    gt = batch[i].copy()
                    fake = traj[i]

                    fake[:, 0:2] = np.cumsum(fake[:, 0:2], axis=0)
                    fake = clear_traj(fake)
                    fake = down_sample(fake)
                    fake = center_and_normalize(fake)
                    #print(fake)
                    #draw(fake, i=i + 10*r, fake=True)

                    gt[:, 0:2] = np.cumsum(gt[:, 0:2], axis=0)
                    gt = clear_traj(gt)
                    gt = down_sample(gt)
                    gt = center_and_normalize(gt)
                    #print(gt)
                    #draw(gt, i=i, fake=False)

                    d = norm_dtw(fake[:,:2], gt[:,:2])
                    #print(f'{i}:  {d}', len(fake), len(gt))
                    norm_d += d
                    
                norm_d /= bs
                norm_ds.append(norm_d)

            print(f'{round+1} roudn in total 60,  norm_d: {norm_d}')
            print(f'### current dtw : {np.mean(norm_ds)}###')



if __name__ == '__main__':
    ### load model ###
    print('####### create ddpm models ########')
    ddpm = DDPM()
    path = './models_con/unetddpm90000.pth'
    ddpm.load_state_dict(torch.load(path, map_location='cpu'))
    ddpm.to(device)
    print('####### create ddpm models successfully! ########')

    ### load style ###
    print('####### create style models ########')
    style_en = style_encoer()
    style_en.load_state_dict(torch.load('./models_con/style_encoder90000.pth', map_location='cpu'))
    style_en.to(device)
    print('####### create style models successfully! ########')

    ### load char classifier ###
    print('####### create classifier models ########')
    char_classifier = char_classifier_testset()
    char_classifier.load_state_dict(torch.load("./evaluate_char/char_classifier_model/classifier20000.pth", map_location='cpu'))
    char_classifier.to(device)
    classifier_ = char_classifier_testset()
    classifier_.load_state_dict(torch.load("./evaluate_char/char_classifier_model/classifier30000.pth", map_location='cpu'))
    classifier_.to(device)
    print('####### create char classifier models successfully! ########')

    ### load style classifier ###
    print('####### create style classifier models ########')
    style_classifier = style_encoder_testset()
    style_classifier.load_state_dict(torch.load("./evaluate_char/style_classifier_model/style_classifier11000.pth", map_location='cpu'))
    style_classifier.to(device)
    print('####### create style classifier models successfully! ########')

    evaluate_socre(ddpm, style_en, char_classifier, style_classifier, classifier_)
    #evaluate_dtw(ddpm, style_en, char_classifier, style_classifier, classifier_)
    print(path)