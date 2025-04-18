import torch.optim as optim
from style_classifier import *
from unet_ddpm import *
#from char_classifier import char_classifier

char_datas = np.load('/lustre/home/msren/database/char/datas.npy', allow_pickle=True).tolist()
mydict = np.load('/lustre/home/msren/database/char/mydict.npy', allow_pickle=True).tolist()
dict_size = len(mydict)


###  学习率  ############
def lr_decay(optimizer):
    # Decay learning rate by a factor of lr_decay
    for param_group in optimizer.param_groups:
        if param_group['lr'] > hp.min_lr:
            param_group['lr'] *= hp.lr_decay
    return optimizer


def draw(traj, i=0, fake=True):
    seq = traj.copy()  # 不改变原序列
    seq[:, 0:2] = np.cumsum(traj[:, 0:2], axis=0)
    strokes = np.split(seq, np.where(seq[:, 2] == -1)[0] + 1)  # split[x,y]到x前一个截止
    strokes.pop()
    for s in strokes:
        plt.plot(s[:, 0], -s[:, 1], linewidth=1.5)

    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()
    if fake:
        plt.savefig('./img/test{}.png'.format(i),dpi=700,bbox_inches="tight")
    else:
        plt.savefig('./img/true{}.png'.format(i),dpi=700,bbox_inches="tight")
    plt.close()


def pad(line, k=8):
    # 使line  b l 3 整除k
    b, l, _ = line.shape
    if l % k == 0:
        return line
    else:
        pad_l = k - (l % k)
        tensor = torch.tensor([0,0,-1]).unsqueeze(0).repeat(b, pad_l, 1).to(device)
        line = torch.cat([line, tensor], dim=1)

        return line


def parser_label(label):
    ## 输入一个label list 返回其中每个字符的长度
    duration = []
    l = 0
    total = len(label)

    for k in range(total):
        if label[k] == 4052: # EOS
            duration.append(total - k)
            return duration
        elif label[k] == 4054: # End
            l += 1
            duration.append(l)
            l = 0
        else:
            l += 1

    return duration


def get_batch(bs = 64):
    labels = []
    lens = []
    lines = np.zeros((bs, 120, 3))
    ids = np.random.choice(1020, bs, replace=False).tolist()

    for i in range(bs):
        id = ids[i]
        char_data = char_datas[id]
        num = len(char_data)
        j = np.random.randint(0, num)
        char = char_data[j][0]
        char_id = mydict.index(char)
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
    
    max_len = max(lens)
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
    
    return batch, labels, durations, style_refs


def test_model(model, style_en, classifier=None):
    model.eval()
    with torch.no_grad():
        batch, labels, durations, style_refs = get_batch(bs=10)
        print(durations)
        style_refs = style_refs.permute(0, 2, 1)
        x, en1, en2, en3 = style_en(style_refs)
        style_features = [x, en1, en2, en3]

        ### 检测数据是否正确
        traj = batch.detach().cpu().numpy()
        for i in range(10):
            txt = ''
            for id in labels[i]:
                txt += mydict[id]
            print(txt)
            draw(traj[i], i=i, fake=False)
        print(traj.shape)
        print('finished')

        ### 生成 可以多生成几次
        for r in range(3):
            if classifier == None:
                traj = ddpm.generate(labels, durations, style_features).permute(0, 2, 1) #b,num,_tag,3
            else:
                traj = ddpm.guided_generate(labels, durations, classifier, style_features).permute(0, 2, 1) #b,num,_tag,3

            traj[:, :, 2] = torch.sgn(traj[:, :, 2])
            traj = traj.cpu().numpy()
            for i in range(10):
                draw(traj[i], i=i+10*r, fake=True)
            print(traj.shape)
            print('finished')


def train_model(model, style_en, style_proj, epochs=hp.epo):
    model.train()
    optimizer = optim.Adam(list(model.parameters())+list(style_en.parameters())+list(style_proj.parameters()), lr=hp.lr)

    for epoch in range(epochs):
        print(f'#### batch : {epoch}')

        batch, labels, durations, style_refs = get_batch()
        style_refs = style_refs.permute(0, 2, 1)
        x, en1, en2, en3 = style_en(style_refs)
        # print(x.shape)      64, 128, 1000
        # print(en1.shape)    64, 256, 500
        # print(en2.shape)    64, 512, 250
        # print(en3.shape)    64, 1024, 125

        style_features = [x, en1, en2, en3]
        loss_c = style_proj.contrastive_loss_all(en1, en2, en3)
        loss_r = model.loss_function(batch, labels, durations, style_features)

        loss = loss_c + loss_r
        print(f'loss:{loss.item()}   loss_r:{loss_r.item()}    loss_c:{loss_c.item()} \n')
        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(list(model.parameters())+list(style_en.parameters())+list(style_proj.parameters()), hp.grad_clip)
        optimizer.step()
        optimizer = lr_decay(optimizer)

        # save model
        if epoch%10000 == 0 and epoch > 1:
            ### 检测数据是否正确
            with torch.no_grad():
                traj = batch.detach().cpu().numpy()
                for i in range(10):
                    txt = ''
                    for id in labels[i]:
                        txt += mydict[id]
                    #print(txt)
                    draw(traj[i], i=i, fake=False)
                print(traj.shape)
                print('finished')

                ### 生成 可以多生成几次
                for r in range(1):
                    traj = ddpm.generate(labels, durations, style_features).permute(0, 2, 1) #b,num,_tag,3
                    traj[:, :, 2] = torch.sgn(traj[:, :, 2])
                    traj = traj.cpu().numpy()
                    for i in range(10):
                        draw(traj[i], i=i+10*r, fake=True)
                    print(traj.shape)
                    print('finished')
            
            if loss_r.item() < 0.02 or (epoch%50000 == 0 and epoch > 1): 
                print('------saving model-----')
                torch.save(model.state_dict(), './models_con/unetddpm{}.pth'.format(epoch+50000))
                torch.save(style_en.state_dict(),'./models_con/style_encoder{}.pth'.format(epoch+50000))
                torch.save(style_proj.state_dict(),'./models_con/style_projector{}.pth'.format(epoch+50000))


if __name__ == '__main__':
    ### load model ###
    print('####### create ddpm models ########')
    ddpm = DDPM()
    ddpm.load_state_dict(torch.load('./models_con/unetddpm50000.pth', map_location='cpu'))
    ddpm.to(device)
    print('####### create ddpm models successfully! ########')

    ### load style ###
    print('####### create style models ########')
    style_en = style_encoer()
    style_en.load_state_dict(torch.load('./models_con/style_encoder50000.pth', map_location='cpu'))
    style_en.to(device)

    style_proj = style_projector()
    style_proj.load_state_dict(torch.load('./models_con/style_projector50000.pth', map_location='cpu'))
    style_proj.to(device)
    print('####### create style models successfully! ########')

    '''print('####### create char_classifier models ########')
    classifier = char_classifier()
    classifier.load_state_dict(torch.load("./char_classifier_model/classifier_down40000.pth", map_location='cpu'))
    classifier.to(device)
    print('####### create char_classifier models successfully! ########')'''

    ddpm, style_en = train_model(ddpm, style_en, style_proj)
    #test_model(ddpm, style_en)