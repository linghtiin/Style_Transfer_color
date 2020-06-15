import os
from tqdm import tqdm
from pathlib import Path
import torch as th
import numpy as np
import pandas as pd
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from Source.myModels import Models
from Source.myModels.dataset import PreprocessDataset
from Source.myTools import TrainTools


def train(args):
    dir_image, dir_loss, dir_model_state = TrainTools.mk_train_dir(args.save_dir)
    device = TrainTools.check_gpu(args.gpu)

    print(f'# Minibatch-size: {args.batch_size}')
    print(f'# epoch: {args.epoch}')
    print('')

    # prepare dataset and dataLoader
    train_dataset = PreprocessDataset(args.train_content_dir, args.train_style_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    iters_num = round(len(train_dataset) / args.batch_size)
    print(f'Length of train image pairs: {iters_num}')

    model = TrainTools.load_model(args, device)

    optimizer = Adam(model.parameters(), lr=args.learning_rate, betas=args.betas)

    # start training
    loss_list = []
    test_gen = TrainTools.log_test(args.test_content_dir, args.test_style_dir, model, device)
    _flag = True
    for epoch in range(1, args.epoch + 1):
        print(f'Start {epoch} epoch')
        if epoch % 10 == 0:
            _flag = True

        for step, (content, style) in enumerate(train_loader, 1):
            _total_step = (epoch - 1) * iters_num + step
            content = content.to(device)
            style = style.to(device)
            loss = model(content, style, alpha=1.0, lam=args.lams)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            print(f'[{epoch}/total {args.epoch} epoch],[{step} /'
                  f'total {iters_num} iteration]: {loss.item()}')

            if step % args.snapshot_interval == 0 and _flag:
                try:
                    img_gen = next(test_gen)
                    save_image(img_gen, f'{dir_image}/{epoch}_epoch_{step}_iteration.png', nrow=args.batch_size)
                except StopIteration:
                    test_gen = TrainTools.log_test(args.test_content_dir, args.test_style_dir, model, device)
                    _flag = False

    th.save(model.state_dict(), f'{dir_model_state}/{args.model_path}.pth')

    loss_mean = TrainTools.mk_loss_plt(dir_loss, loss_list)
    return loss_mean


class Parser1:
    def __init__(self):
        """
        - 请保证model与reuse匹配。

        """
        self.batch_size = 2
        self.epoch = 20
        self.gpu = 0
        self.lams = [10, 20, 30, 40, 60, 60]
        self.learning_rate = 5e-5
        self.betas = (0.5, 0.999)
        self.snapshot_interval = 5
        self.train_content_dir = '.dataset/My_dataset/content'
        self.train_style_dir = '.dataset/My_dataset/style'
        self.test_content_dir = '.dataset/My_dataset/content'
        self.test_style_dir = '.dataset/My_dataset/style'
        self.save_dir = '.result/adaIn_result'
        self.model = None
        self.reuse = None
        self.noAdaIN = False
        self.isProColor = True
        self.model_path = 'first_epoch'


def use_train01():
    cmd_args = Parser1()
    # cmd_args.noAdaIN = True
    # cmd_args.isProColor = False
    # cmd_args.epoch = 2
    # cmd_args.snapshot_interval = 1  # 输出测试图像

    # cmd_args.train_content_dir = '.dataset/val2017'

    # 经过修改后的模型
    cmd_args.model = Models.Model9_color2
    cmd_args.reuse = '.result/color2_result/range1_epoch.pth'

    cmd_args.save_dir = '.result/_result'
    cmd_args.model_path = 'range2_epoch'
    loss_t = train(cmd_args)
    print(loss_t)


def use_train_main():
    cmd_args = Parser1()
    cmd_args.noAdaIN = True
    cmd_args.isProColor = False
    cmd_args.epoch = 200
    # cmd_args.lams = [20, 30, 30, 40, 60, 60]
    cmd_args.batch_size = 2
    cmd_args.learning_rate = 0.00003
    # cmd_args.betas = (0.5, 0.9)
    cmd_args.snapshot_interval = 1  # 输出测试图像

    # cmd_args.train_content_dir = '.dataset/val2017'

    # 原始论文的预训练模型
    # cmd_args.model = Models.Model
    # cmd_args.reuse = '.models/pre_model_state.pth'
    # cmd_args.save_dir = '.result/_result'

    # 经过修改后的模型
    cmd_args.model = Models.Model15_color3
    cmd_args.save_dir = '.result/color3_result'

    train(cmd_args)

    for i in range(1, 4):
        cmd_args.reuse = cmd_args.save_dir + f'/{cmd_args.model_path}.pth'
        cmd_args.model_path = 'range%d_epoch' % i

        if i == 1:
            cmd_args.train_content_dir = '.dataset/val2017'
        elif i == 2:
            cmd_args.noAdaIN = False
        elif i == 3:
            cmd_args.isProColor = True
        train(cmd_args)


if __name__ == '__main__':
    os.chdir('./../')

    # use_train01()
    use_train_main()
