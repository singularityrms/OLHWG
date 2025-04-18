from basemodel import *

class HParams():
    def __init__(self):
        # Encoder
        # Decoder
        # 训练超参数
        self.bs = 64
        self.epo = 2000001
        self.lr = 0.0001
        self.min_lr = 0.00001
        self.lr_decay = 0.9995
        self.grad_clip = 1.

        # DDPM
        self.time_emb_size = 16
        self.char_emb_size = 256
        self.time_step = 1000 #200
        self.beta_0 = 1e-4  #0.002
        self.beta_T = 2*1e-2 #0.1


hp = HParams()

#### cross attention ####
class Cross_attention(nn.Module):
    def __init__(self, dim1, dim2, q_dim, k_dim, v_dim):
        super().__init__()
        self.q_func = nn.Linear(dim1, q_dim)
        self.k_func = nn.Linear(dim2, k_dim)
        self.v_func = nn.Linear(dim2, v_dim)

        self.out_func = nn.Linear(v_dim, dim1)

    def forward(self, x, f):
        x = x.permute(0, 2, 1) # b L dim1
        f = f.permute(0, 2, 1) # b L dim2
        q = self.q_func(x)
        k = self.k_func(f)
        v = self.v_func(f)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) / (256 ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        output = torch.bmm(attn_weights, v)
        output = self.out_func(output)
        output += x
        output = output.permute(0, 2, 1)

        return output

#### network #######
class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, dim=128):
        super().__init__()
        self.en_block1 = EncoderBlock(in_channels, dim)
        self.en_block2 = EncoderBlock(dim*2, dim*2)
        self.en_block3 = EncoderBlock(dim*4, dim*4)
        self.en_block4 = EncoderBlock(dim*8, dim*8)
        self.downblock1 = EncoderBlock(dim, dim*2, 2)
        self.downblock2 = EncoderBlock(dim*2, dim*4, 2)
        self.downblock3 = EncoderBlock(dim*4, dim*8, 2)

        self.de_block1 = EncoderBlock(dim*4, dim*4)
        self.de_block2 = EncoderBlock(dim*2, dim*2)
        self.de_block3 = EncoderBlock(dim, in_channels)
        self.de_block4 = EncoderBlock(in_channels, out_channels)
        self.upblock1 = DecoderBlock(dim*8, dim*4, 2)
        self.upblock2 = DecoderBlock(dim*4, dim*2, 2)
        self.upblock3 = DecoderBlock(dim*2, dim, 2)

        #self.cross_attention1 = Cross_attention(dim, 128, 256, 256, 256)
        self.cross_attention1 = Cross_attention(in_channels, 128, 256, 256, 256)
        self.cross_attention2 = Cross_attention(dim*2, 256, 256, 256, 256)
        self.cross_attention3 = Cross_attention(dim*4, 512, 256, 256, 256)
        self.cross_attention4 = Cross_attention(dim*8, 1024, 256, 256, 256)

        self.elu = nn.ELU()

    def forward(self, x, style_features=None):
        # encoder
        en1 = self.elu(self.en_block1(x))
        #en1 = self.cross_attention1(en1, style_features[0])
        en2 = self.elu(self.downblock1(en1))

        en3 = self.elu(self.en_block2(en2))
        #en3 = self.cross_attention2(en3, style_features[1])
        en4 = self.elu(self.downblock2(en3))

        en5 = self.elu(self.en_block3(en4))
        #en5 = self.cross_attention3(en5, style_features[2])
        en6 = self.elu(self.downblock3(en5))

        # decoder
        de1 = self.elu(self.en_block4(en6))
        de1 = self.cross_attention4(de1, style_features[3])
        de2 = self.elu(self.upblock1(de1))

        de3 = self.elu(self.de_block1(de2 + en5))
        de3 = self.cross_attention3(de3, style_features[2])
        de4 = self.elu(self.upblock2(de3 + en4))

        de5 = self.elu(self.de_block2(de4 + en3))
        de5 = self.cross_attention2(de5, style_features[1])
        de6 = self.elu(self.upblock3(de5 + en2))

        de7 = self.elu(self.de_block3(de6 + en1))
        de7 = self.cross_attention1(de7, style_features[0])
        output = self.de_block4(de7)

        return output
    

#### DDPM ####
class DDPM(nn.Module):
    def __init__(self, input_channel=hp.time_emb_size + hp.char_emb_size + 4, output_channel=3):
        super(DDPM, self).__init__()
        self.time_emb = SinusoidalPositionEmbeddings(hp.time_emb_size)
        self.denoiser = Unet(input_channel, output_channel)

        ## DDPM 超参数
        self.beta = cosine_schedule(hp.beta_T, hp.beta_0, hp.time_step).to(device)
        self.alpha = 1 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, -1).to(device)

        ## 文字编码字典
        dict_size = 4055
        self.char_embeddings = nn.Embedding(dict_size, hp.char_emb_size)

        ## loss
        self.loss = nn.MSELoss()

    def get_positions(self, durations):
        positions = []
        for duration in durations:
            concatenated_partitions = []
            for d in duration:
                partition = torch.linspace(0, 1, d)
                concatenated_partitions.append(partition)
            concatenated_partitions = torch.cat(concatenated_partitions).unsqueeze(0).unsqueeze(0) #1,1,length
            positions.append(concatenated_partitions)

        batch_positions = torch.cat(positions).to(device) # b, 1, length

        return batch_positions

    def get_contents_times(self, inputs, t, tags, durations):
        # tags: b, 500   return: b, dim, 500
        # inputs : b, 64, 500 + contents + time_emb ---- b 256 500
        # t : tensor

        #### content embeddings ####
        tags = torch.from_numpy(np.array(tags)).to(device)
        content_features = self.char_embeddings(tags).permute(0, 2, 1)
        bs, dim, L = content_features.size()

        #### time embeddings ####
        bs, _, length = inputs.shape
        inputs_t = self.time_emb(t).unsqueeze(2)  # 1, dim, 1
        if inputs_t.shape[0] != bs:
            inputs_t = inputs_t.repeat(bs, 1, length)
        else:
            inputs_t = inputs_t.repeat(1, 1, length)

        #### postion emb for every char ####
        batch_positions = self.get_positions(durations) # 
        #print(batch_positions)
        inputs = torch.cat([inputs, batch_positions, content_features, inputs_t], 1)

        return inputs

    def pred_noise(self, inputs, style_features=None):
        #预测噪声
        outputs = self.denoiser(inputs, style_features)
        return outputs

    def sample_x_t(self, x, t):
        noise = torch.randn_like(x).to(device)
        x_t = torch.zeros_like(x).to(device)
        #print(x.shape, t.shape)
        if t.shape[0] == 1:
            x_t = torch.sqrt(self.alpha_cumprod[t]) * x + torch.sqrt(1 - self.alpha_cumprod[t]) * noise
            return x_t, noise

        for i in range(x.shape[0]):
            x_t[i] = torch.sqrt(self.alpha_cumprod[t[i]]) * x[i] + torch.sqrt(1 - self.alpha_cumprod[t[i]]) * noise[i]
        return x_t, noise
    
    def remove_noise(self, x_t, t, predict_noise):
        # 返回x_t-1 的均值
        temp = x_t - (self.beta[t] / torch.sqrt(1 - self.alpha_cumprod[t])) * predict_noise
        return temp / torch.sqrt(self.alpha[t])
    
    def generate(self, tags, durations, style_features=None, var=0.9): 
        # tags: b, len
        bs = len(tags)
        length = len(tags[0])

        char_ids = []
        for i in range(bs):
            char_ids.append(tags[i][1])
        # print(char_ids)

        x = torch.randn(bs, 3, length).to(device)
        for i in range(hp.time_step):
            t = hp.time_step - 1 - i
            if t%300 == 0: 
                print(t)
            t = torch.tensor([t]).to(device)

            inputs = self.get_contents_times(x, t=t, tags=tags, durations=durations)
            predict_noise = self.pred_noise(inputs, style_features)
            x = self.remove_noise(x, t, predict_noise)
            
            if t > 15:
               sigma = torch.sqrt(0.8 * self.beta[t] * (1 - self.alpha_cumprod[t-1]) / (1 - self.alpha_cumprod[t]))
               x += sigma * torch.randn_like(x)        

        return x
    
    def loss_function(self, x, tags, durations, style_features=None):
        loss = torch.Tensor([0.]).to(device)
        x = x.permute(0, 2, 1) # b channel L
        bs = x.shape[0]

        for i in range(12):
            t = torch.randint(0, hp.time_step, (bs,)).to(device)
            #t = torch.randint(0, 100, (1,)).to(device)
            if t.shape[0] == 1:
                print(f'time step: {t.item()}')
            x_t, noise = self.sample_x_t(x, t)
            x_t = self.get_contents_times(x_t, t, tags, durations)
            predict_noise = self.pred_noise(x_t, style_features)
            loss += self.loss(predict_noise, noise)
        
        loss /= 12
        return loss

    #### 单字生成的classifier guidance ####
    def optimize_xt(self, xt, classifier, char_ids, lr):
        xt.requires_grad_(True)

        p = classifier(xt)
        p = torch.softmax(p, dim=1)
        p = p[np.arange(len(char_ids)), char_ids].mean()
        loss = -p
        loss.backward()

        with torch.no_grad():
            #print(lr * xt.grad)
            xt -= lr * xt.grad

        return xt

    def guided_generate(self, tags, durations, classifier=None, style_features=None): 
        # tags: b, len
        bs = len(tags)
        length = len(tags[0])

        char_ids = []
        for i in range(bs):
            char_ids.append(tags[i][1])
        #print(char_ids)

        x = torch.randn(bs, 3, length).to(device)
        for i in range(hp.time_step):
            t = hp.time_step - 1 - i
            if t%300 == 0: 
                print(t)
            t = torch.tensor([t]).to(device)

            inputs = self.get_contents_times(x, t=t, tags=tags, durations=durations)
            predict_noise = self.pred_noise(inputs, style_features)
            x = self.remove_noise(x, t, predict_noise)
            
            if t > 15:
                sigma = torch.sqrt(0.8 * self.beta[t] * (1 - self.alpha_cumprod[t-1]) / (1 - self.alpha_cumprod[t]))
                x += sigma * torch.randn_like(x)

            #if t > 15 and t < 200:
            #    with torch.enable_grad():
            #        x = self.optimize_xt(x, classifier, char_ids, lr=0.1)

        return x