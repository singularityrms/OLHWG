# -*- coding: utf-8 -*-
import torch.optim as optim
from basemodel import *
from tqdm import tqdm
import torch.nn.functional as F

##### style classifier ######
class style_encoer(nn.Module):
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
        self.func_c = nn.Linear(1024, 1020)
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


class style_projector(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.project1 = nn.Linear(2*dim, 2*dim)
        self.project2 = nn.Linear(4*dim, 4*dim)
        self.project3 = nn.Linear(8*dim, 8*dim)

        self.logit_scale = 2.
        self.w1 = 0.01
        self.w2 = 0.1
        self.w3 = 0.1
    
    def contrastive_loss_all(self, en1, en2, en3):
        # en : batch, D, Length
        en1 = self.project1(en1.permute(0,2,1))
        en2 = self.project2(en2.permute(0,2,1))
        en3 = self.project3(en3.permute(0,2,1))

        loss1 = self.contrastive_loss(en1)
        loss2 = self.contrastive_loss(en2)
        loss3 = self.contrastive_loss(en3)

        print(f'loss1:{loss1.item()}   loss2:{loss2.item()}    loss3:{loss3.item()} \n')
        loss = self.w1 * loss1 + self.w2 * loss2 + self.w3 * loss3

        return loss

    def contrastive_loss(self, en):
        # en : batch, Length, D
        batch_size = en.shape[0]
        features = en[:, :50, :].mean(dim=1)
        features_ = en[:, 50:100, :].mean(dim=1)

        sim_matrix = torch.mm(features, features_.t()) * self.logit_scale
        label = torch.arange(batch_size).to(sim_matrix).long()
        
        # nll loss
        loss_1 = F.nll_loss(F.log_softmax(sim_matrix, dim=1), label)
        loss_2 = F.nll_loss(F.log_softmax(sim_matrix.t(), dim=1), label)
        loss = loss_1 + loss_2

        return loss
    

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

def get_style_batch_10(char_datas, style_ids):
    bs = len(style_ids)
    batch = np.zeros((bs, 1000, 3))

    for i in range(bs):
        id = style_ids[i]
        data = char_datas[id]
        char_num = len(data)
          
        char_ids = np.random.choice(char_num, 15, replace=False)
        temp_data = np.zeros((1000, 3))
        total_length = 0
        for j, char_id in enumerate(char_ids):
            traj = data[char_id][1]
            length = traj.shape[0]
            if total_length + length > 1000:
                remaining_length = 1000 - total_length
                temp_data[start_idx:start_idx+remaining_length, :] = traj[:remaining_length]
                total_length = 1000
                break
            start_idx = total_length
            end_idx = start_idx + length
            temp_data[start_idx:end_idx, :] = traj
            total_length += length

        if total_length < 1000:
            temp_data[total_length:, 2] = -1.
        batch[i] = temp_data
    batch = torch.from_numpy(batch).float().to(device)
    return batch


'''def get_style_batch_10(char_datas, style_ids):
    bs = len(style_ids)
    batch = np.zeros((bs, 1200, 3))

    for i in range(bs):
        id = style_ids[i]
        data = char_datas[id]
        char_num = len(data)      
        char_ids = np.random.choice(char_num, 10, replace=False)
        temp_data = np.zeros((1200, 3))
        end_idx = 0
        for j, char_id in enumerate(char_ids):
            traj = data[char_id][1]
            length = traj.shape[0]
            start_idx = end_idx
            end_idx = start_idx + length
            temp_data[start_idx:end_idx, :] = traj
        temp_data[end_idx:, 2] = -1.
        
        batch[i] = temp_data
    batch = torch.from_numpy(batch).float().to(device)
    return batch'''

    
def train_model(model, epochs=1000000):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        style_ids = np.random.choice(1020, 300)
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
        if epoch%20000 == 0 and acc>0.9:
            #print('------valid model-----')
            #acc = test_model(model)
            #print(f'acc: {acc}')

            if acc > 0.8:
                print('------saving model-----')
                torch.save(model.state_dict(), './style_encoder_model/style_encoder80{}.pth'.format(epoch))
                return 0
                
    return model

def test_model(model):
    num = 1020
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
    char_datas = np.load('/lustre/home/msren/database/char/datas.npy', allow_pickle=True)
    print('####### create se models ########')
    model = style_encoer()
    #model.load_state_dict(torch.load('/lustre/home/msren/generate_line/SE/cnn_classifier90000.pth', map_location='cpu'))
    print('####### create se models successfully! ########')

    model.to(device)
    model = train_model(model)