# -*- coding: utf-8 -*-
import sys
sys.path.append('..')
import torch.optim as optim
from basemodel import *
from tqdm import tqdm

char_datas = np.load('/lustre/home/msren/database/char/test_datas.npy', allow_pickle=True)
mydict = np.load('/lustre/home/msren/database/char/mydict.npy', allow_pickle=True).tolist()
dict_size = len(mydict)

##### char classifier ####
class char_classifier_testset(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.linear = Conv1d(in_channels=3, out_channels=dim, kernel_size=3)

        self.en_block1 = nn.Sequential(
              EncoderBlock(dim, dim*2, 2))
              
        self.en_block2 = nn.Sequential(
              EncoderBlock(dim*2, dim*4, 2))
              
        self.en_block3 = nn.Sequential(
              EncoderBlock(dim*4, dim*8, 2))

        self.en_block4 = nn.Sequential(
              EncoderBlock(dim*8, dim*16, 2))

        self.elu = nn.ELU()

        ### train ###
        self.func_c = nn.Linear(dim*16, 3755)
        self.cross_entopy = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.linear(x)  #b 128 L
        en1 = self.en_block1(self.elu(x))  # b 256 L/2 
        en2 = self.en_block2(self.elu(en1))
        en3 = self.en_block3(self.elu(en2))
        en4 = self.en_block4(self.elu(en3))
        f = torch.mean(en4, dim=2)  # b 2048
        p = self.func_c(f)  # b  3054

        return p

    def loss_function(self, x, char_ids):
        x = x.permute(0, 2, 1)
        p = self.forward(x)
        loss = self.cross_entopy(p, char_ids)
        # Calculate accuracy
        preds = torch.argmax(p, dim=1)
        correct = torch.sum(preds == char_ids)
        accuracy = float(correct) / len(char_ids)
        print(f'accuracy: {accuracy}')

        return loss, accuracy

    def acc_function(self, x, char_ids):
        x = x.permute(0, 2, 1)
        p = self.forward(x)
        # Calculate accuracy
        preds = torch.argmax(p, dim=1)
        correct = (preds == char_ids)

        return correct


### 学习率 #####
def lr_decay(optimizer):
    for param_group in optimizer.param_groups:
        if param_group['lr'] > 0.00001:
            param_group['lr'] *= 0.9998
    return optimizer

### train #####
def get_char_batch(bs = 512):
    style_ids = np.random.choice(60, bs, replace=True)
    batch = np.zeros((bs, 120, 3))
    ids = []

    for i in range(bs):
        style_id = style_ids[i]
        data = char_datas[style_id]
        char_num = len(data)
        
        char_id =  np.random.choice(char_num, 1, replace=False)[0]
        traj = data[char_id][1]
        length = traj.shape[0] 
        id = mydict.index(data[char_id][0])
        ids.append(id)

        batch[i][:length, :] = traj
        batch[i][length:, 2] = -1.
        
    batch = torch.from_numpy(batch).float().to(device)
    ids = torch.tensor(ids).long().to(device)

    return batch, ids


def train_model(model, epochs=100000):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(epochs):
        batch, ids = get_char_batch()
        loss, acc = model.loss_function(batch, ids)

        optimizer.zero_grad()
        #print("#### batch : {}".format(epoch))
        print("#### batch : {}, loss: {}".format(epoch, loss.item()))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer = lr_decay(optimizer)

        if epoch % 10000 == 0 and acc > 0.97:
            torch.save(model.state_dict(), './char_classifier_model/classifier_down{}.pth'.format(epoch))

    return model


if __name__ == '__main__':
    print('####### create char_classifier models ########')
    model = char_classifier()
    print('####### create char_classifier models successfully! ########')

    model.to(device)
    model = train_model(model)