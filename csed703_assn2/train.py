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
EPOCH = 8

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
    train_loader = DataLoader(
        train_set, shuffle=True,
        collate_fn=lambda li: (torch.LongTensor([li[0][0]]), li[0][1]))

    _, sample_text = train_set[0]
    
    print('input_size: {}'.format(sample_text.size()[2]))
    print('output_clases: {}'.format(train_set.class_names))
    composition = [len(l) for l in train_set.lines_per_class_ind]
    print('dataset_composition: {}'.format(composition))

    predictor_module = model.CityNamePredictor(
        sample_text.size()[2], len(train_set.class_names)).cuda()
    print(predictor_module)
    loss_module = model.CityNameLoss().cuda()
    optim_module = torch.optim.SGD(predictor_module.parameters(), lr=LR)

    cnt = 0
    iters = []
    train_loss = []
    val_loss = []
    running_loss = 0.0
    for epoch in range(EPOCH):
        for i, (label, text) in enumerate(train_loader):
            predictor_module.zero_grad()
            pred = predictor_module.train()(text.cuda()).float()
            loss = loss_module(label.cuda(), pred)
            loss.backward()
            optim_module.step()

            CALLBACK_N = len(train_set)//4
            def callback_every_n(func, *args):
                if cnt % CALLBACK_N == 0:
                    return func(*args)

            running_loss += loss.item()
            def get_running_loss():
                if cnt == 0:
                    return running_loss
                else:
                    return running_loss / CALLBACK_N

            def print_loss():
                print('[{}/{}]Loss : {}'.format(epoch, i, get_running_loss()))

            callback_every_n(print_loss)
            val_result = callback_every_n(val_hook,
                                   val_set, predictor_module, loss_module, cnt)
            if val_result is not None:
                val, val_accuracy = val_result
                iters.append(cnt)
                train_loss.append(get_running_loss())
                val_loss.append(val.item())
                print('Validation accuracy: {}'.format(val_accuracy))
                running_loss = 0.0
                vis_hook('loss_graph.png', iters, train_loss, val_loss)
                torch.save(predictor_module.state_dict(), chk_path)

            cnt += 1

if __name__ == '__main__':
    train('checkpoint/checkpoint.chk', test.test, vis.save_loss_graph)
