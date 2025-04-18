from style_classifier import *
from unet_ddpm import *
from train_ddpm import pad, parser_label
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

char_datas = np.load('/lustre/home/msren/database/char/test_datas.npy', allow_pickle=True)
#char_datas = np.load('/lustre/home/msren/database/char/datas.npy', allow_pickle=True).tolist()
mydict = np.load('/lustre/home/msren/database/char/mydict.npy', allow_pickle=True).tolist()


### bs个人 每个人 num个样本 ###
def get_batch(bs=10, num=20):
    labels = []
    lens = []
    lines = np.zeros((bs*num, 120, 3))

    ids = []
    for b in range(bs):
        ids += [b]*num
    char_ids = []

    for i in range(len(ids)):
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
        lines[i, length:, 2] = -1.

        label = [char_id] * length  
        label[0] = 4053
        label[-1] = 4054
        labels.append(label)
    
    #max_len = max(lens)
    #lines = lines[:, :max_len, :]
    lines = torch.from_numpy(lines).float().to(device)
    batch = pad(lines, 8)
    #style_refs = get_style_batch_10(char_datas, ids)
    style_refs = None

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
    return batch, labels, durations, style_refs, char_ids, style_ids


def tsne_style(style_en):
    bs = 10
    num = 200
    batch, labels, durations, style_refs, char_ids, style_ids = get_batch(bs,num)
    ### 先对真实数据做tsne
    with torch.no_grad():
        x, en1, en2, en3 = style_en(batch.permute(0, 2, 1))
    #x, en1, en2, en3 = style_en(style_refs.permute(0, 2, 1))
    en3 = torch.mean(en3, dim=2) #b 1024
    X = en3.detach().cpu().numpy()

    labels = np.repeat(np.arange(bs), num)
    colors = ['#FF0000', '#FFA500', '#FFFF00', '#008000', '#0000FF','#00BFFF', '#800080', '#FFC0CB', '#808080', '#000000']
    # 使用t-SNE进行降维和可视化
    tsne = TSNE(n_components=2, perplexity=15, learning_rate=0.5, random_state=99, n_iter=50000)
    embedded_data = tsne.fit_transform(X)
    scaler = StandardScaler()
    embedded_data = scaler.fit_transform(embedded_data)
    print(embedded_data.shape)
    # 绘制散点图
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(len(colors)-5):
        ax.scatter(embedded_data[labels==i, 0], embedded_data[labels==i, 1], s=10, color=colors[i], label=f'Writer {i}')
    ax.legend(fontsize=8)
    # 调整标签和图例字体大小
    plt.setp(ax.get_legend().get_texts(), fontsize='8')
    plt.setp(ax.get_legend().get_title(), fontsize='8')
    # 保存图像到本地文件
    plt.savefig('./paper/tsne_style/testset_16.png', dpi=300)
    #plt.savefig('./paper/tsne_style/testset_nocon.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    ### load model ###
    '''print('####### create ddpm models ########')
    ddpm = DDPM()
    path = './models_con/unetddpm90000.pth'
    ddpm.load_state_dict(torch.load(path, map_location='cpu'))
    ddpm.to(device)
    print('####### create ddpm models successfully! ########')'''

    ### load style ###
    print('####### create style models ########')
    style_en = style_encoer()
    style_en.load_state_dict(torch.load('./models_con/style_encoder160000.pth', map_location='cpu'))
    #style_en.load_state_dict(torch.load('./models/style_encoder450000.pth', map_location='cpu'))
    style_en.to(device)
    print('####### create style models successfully! ########')

    tsne_style(style_en)
    #print(path)