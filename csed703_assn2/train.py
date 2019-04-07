'''
Training executable script + training cycle definition
'''
import torch
from torch.utils.data.dataloader import DataLoader

import dataset
import test
import model
import vis

LR = 0.005
EPOCH = 10

def train(chk_path, val_hook, vis_hook):
    """
    Train a model and save model checkpoints

    chk_path        checkpoint path to be usedduring training
                    this file is always overwritten with start of every training
    val_hook        a hook for validation
    vis_hook        a hook for collecting and visualizing training progress and result
    """
    train_set = dataset.FilenameClassDataset(
        'data/cities_train/train', transform=dataset.TextlineToVector())
    val_set = dataset.FilenameClassDataset(
        'data/cities_val/val', transform=dataset.TextlineToVector())
    train_loader = DataLoader(train_set, shuffle=True)

    _, sample_text = train_set[0]

    predictor_module = model.CityNamePredictor(
        sample_text.size()[2], len(train_set.class_names)).cuda()
    loss_module = model.CityNameLoss().cuda()
    optim_module = torch.optim.Adam(predictor_module.parameters(), lr=LR)

    cnt = 0
    iters = []
    train_loss = []
    val_loss = []
    for epoch in range(EPOCH):
        for i, (label, text) in enumerate(train_loader):
            predictor_module.zero_grad()
            pred = predictor_module.train()(text.cuda())
            loss = loss_module(torch.Tensor(label).cuda(), pred.cuda())
            loss.backward()
            optim_module.step()

            cnt += 1

            def callback_every_n(n, func, *args):
                if cnt % n == 1:
                    return func(*args)

            def print_loss():
                print('[{}/{}]Loss : {}'.format(epoch, i, loss.item()))

            callback_every_n(20, print_loss)
            val = callback_every_n(len(train_set)//2, val_hook,
                                   val_set, predictor_module, loss_module)
            if val is not None:
                iters.append(cnt)
                train_loss.append(loss.item())
                val_loss.append(val.item())
        vis_hook('loss_graph.png', iters, train_loss, val_loss)
        torch.save(predictor_module.state_dict(), chk_path)

if __name__ == '__main__':
    train('checkpoint/checkpoint.chk', test.test, vis.save_loss_graph)
