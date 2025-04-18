from os import write
from random import randrange
from generate_line import *


def generate_test_lines(ddpm, style_en, box_genertaor):
    all_datas = []
    with open('./evaluate_line/info.txt','r') as f:
        for writer_id in range(1000,1050):
            data_one_writer = []
            for j in range(1):
                print(writer_id, j)
                traj, text = generate_line(1040, ddpm, style_en, box_genertaor)
                data_one_writer.append(traj)

                last_column = traj[:, -1]
                last_column[last_column == -1] = 0
                traj[:, -1] = last_column
                #np.savetxt(f'./evaluate_line/{writer_id}_{j}.txt', traj, fmt='%.6f', delimiter=' ')
                #f.write(f'{writer_id}_{j}.txt     {text}\n')

        all_datas.append(data_one_writer)

    f.close()
    np.save('./evaluate_line/all_datas.npy', all_datas)


if __name__ == '__main__':
    ### load model ###
    print('####### create models ########')
    ddpm = DDPM()
    ddpm.load_state_dict(torch.load('./models/unetddpm290000.pth', map_location='cpu'))
    ddpm.to(device)
    style_en = style_encoer()
    style_en.load_state_dict(torch.load('./models/style_encoder290000.pth', map_location='cpu'))
    style_en.to(device)
    box_genertaor = Box_Lstm(device)
    box_genertaor.load_state_dict(torch.load('./bonibox_gen/boxlstm3000.pth', map_location='cpu'))
    box_genertaor.to(device)
    print('####### create models successfully! ########')

    generate_test_lines(ddpm, style_en, box_genertaor)

    #traj, text = generate_line(1040, ddpm, style_en, box_genertaor)
    #traj[:, -1][traj[:, -1] == -1] = 0
    #np.savetxt(f'./evaluate_line/1040_0.txt', traj, fmt='%.6f', delimiter=' ')