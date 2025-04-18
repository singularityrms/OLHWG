# -*- coding: utf-8 -*-
import sys
sys.path.append('..')
import torch.optim as optim
from basemodel import *
from tqdm import tqdm

##### style classifier ######
class style_encoder_testset(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.linear = Conv1d(in_channels=3, out_channels=dim, kernel_size=3)
        
        self.en_block1 = nn.Sequential(
              EncoderBlock(dim, dim*2, 2))
              
        self.en_block2 = nn.Sequential(
              EncoderBlock(dim*2, dim*4, 2))
              
        self.en_block3 = nn.Sequential(
              EncoderBlock(dim*4, dim*8, 2))
        self.elu = nn.ELU()

        ### train ###
        self.func_c = nn.Linear(1024, 60)
        self.cross_entopy = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.linear(x) # b 128 L
        en1 = self.en_block1(self.elu(x))  # b 256 L/2 
        en2 = self.en_block2(self.elu(en1))
        en3 = self.en_block3(self.elu(en2))

        return x, en1, en2, en3
    
    def loss_function(self, x, style_ids):
        x = x.permute(0, 2, 1)
        _,_,_, f = self.forward(x) # b 1024 12
        f = torch.mean(f, dim=2) # b 1024
        p = self.func_c(f) # b 1020

        loss = self.cross_entopy(p, style_ids)

        # 计算准确率
        _, predicted = torch.max(p.data, 1)
        total = style_ids.size(0)
        correct = (predicted == style_ids).sum().item()
        accuracy = correct / total
        print('acc: ', accuracy)

        return loss, accuracy


    def acc_function_(self, x, style_ids, num):
        x = x.permute(0, 2, 1)
        bs = x.shape[0]
        bs_ = int(bs/num)
        dim = x.shape[1]
        x_reshaped = x.view(bs_, num, dim, -1)
        x_transposed = x_reshaped.transpose(1, 2).contiguous().view(bs_, dim, -1)

        _,_,_, f = self.forward(x_transposed)
        f = torch.mean(f, dim=2) # b 1024
        p = self.func_c(f) # b class

        # Calculate accuracy
        preds = torch.argmax(p, dim=1)
        style_ids_ = style_ids[:bs_]
        correct = (preds == style_ids_)

        return correct
    

    def acc_function(self, x, style_ids):
        x = x.permute(0, 2, 1)
        _,_,_, f = self.forward(x) # b 1024 12
        f = torch.mean(f, dim=2) # b 1024
        p = self.func_c(f) # b 1020

        # Calculate accuracy
        preds = torch.argmax(p, dim=1)
        correct = (preds == style_ids)

        return correct

###  学习率  ############
def lr_decay(optimizer):
    # Decay learning rate by a factor of lr_decay
    for param_group in optimizer.param_groups:
        if param_group['lr'] > 0.00001:
            param_group['lr'] *= 0.9998
    return optimizer

#### train #####
def get_style_batch(style_ids):
    bs = len(style_ids)
    batch = np.zeros((bs, 120, 3))

    for i in range(bs):
        id = style_ids[i]
        data = char_datas[id]
        char_num = len(data)
        char_id =  np.random.choice(char_num, 1, replace=False)
        char_id = char_id[0]
        traj = data[char_id][1]
        length = traj.shape[0] 

        batch[i][:length, :] = traj
        batch[i][length:, 2] = -1.
        
    batch = torch.from_numpy(batch).float().to(device)

    return batch


def train_model(model, epochs=100000):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        style_ids = np.random.choice(60, 128)
        batch = get_style_batch(style_ids)
        style_ids = torch.from_numpy(style_ids).long().to(device)
        loss, acc = model.loss_function(batch, style_ids)

        optimizer.zero_grad()
        print("#### batch : {}, loss: {}".format(epoch, loss.item()))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer = lr_decay(optimizer)

        # save model
        if epoch%1000 == 0 and acc>0.96:
            print('------saving model-----')
            torch.save(model.state_dict(), './style_classifier_model/style_classifier{}.pth'.format(epoch))
                
    return model


def test_model(model):
    num = 60
    total = 0
    count = 0
    for i in tqdm(range(num), desc="Processing"):
        data = char_datas[i]
        char_num = len(data)
        char_num = min(char_num, 300)
        for j in range(char_num):
            char = torch.from_numpy(data[j][1]).float().unsqueeze(0)
            char = char.to(device)
            char = char.permute(0, 2, 1)
            _,_,_, f = model.forward(char)  # b 1024 l
            f = torch.mean(f, dim=2)  # b 1024
            p = model.func_c(f)  # b 1020

            _, pred = torch.max(p, 1)
            pred = pred.detach().item()
            if pred == i:
                count += 1
            total += 1
    acc = count/total

    print(f'{count}/{total}:{acc}')
    return acc


if __name__ == '__main__':
    char_datas = np.load('/lustre/home/msren/database/char/test_datas.npy', allow_pickle=True)
    print('####### create se models ########')
    model = style_encoder_testset()
    print('####### create se models successfully! ########')

    model.to(device)
    model = train_model(model)