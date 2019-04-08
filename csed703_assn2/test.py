'''
Test/validation cycle definition
'''
import os
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

import dataset
import vis
import model

def get_average_loss(predictor_module, loss_module, loader):
    "Get average loss of the whole test dataset"
    loss = 0.0
    cnt = 0
    for label, text in loader:
        pred = predictor_module.eval()(text.cuda())
        loss += loss_module(label.cuda(), pred)
        cnt += 1
    return loss / cnt

def get_confusion_matrix(predictor_module, loader):
    "Get confusion matrix on the test dataset"
    num_class = len(loader.dataset.class_names)

    mat = np.zeros((num_class, num_class), dtype=np.int32)
    for data in loader:
        label, text = data
        pred = predictor_module.eval()(text.cuda()).topk(1)[1][0].item()
        mat[pred, label] += 1

    return mat

def run_confusion(it, predictor, loader):
    mat = get_confusion_matrix(predictor, loader)
    acc = (np.diag(mat).sum() / mat.sum()).item()
    vis.save_confusion_matrix(
        '{}_confusion.png'.format(it),
        loader.dataset.class_names, mat)
    return acc

def test(test_set, predictor_module, loss_module,
         test_id=0,
         loss_hook=get_average_loss,
         vis_hook=run_confusion):
    """
    Test model performance as accuracy

    This function can also be used for validation

    test_set        a dataset.FilenameClassDataset object
    model           loss calculation model
    loss_hook       a hook for calculating loss value.
                    takes 'model', dataloader  and returns loss
    vis_hook        a hook for visualizing test results
    """
    test_loader = DataLoader(
        test_set,
        collate_fn=lambda li: (torch.LongTensor([li[0][0]]), li[0][1]))

    loss = loss_hook(predictor_module, loss_module, test_loader)
    acc = vis_hook(test_id, predictor_module, test_loader)

    return loss, acc

def run_test():
    checkpoint = 'checkpoint/checkpoint.chk'
    if not os.path.exists(checkpoint):
        raise 'Checkpoint does not exist: {}'.format(checkpoint)
    loss_module = model.CityNameLoss()
    test_set = dataset.FilenameClassDataset(
        'data/cities_val/val', transform=dataset.TextlineToVector())

    _, sample_text = test_set[0]
    predictor_module = model.CityNamePredictor(
        sample_text.size()[2], len(test_set.class_names)).cuda()
    chk = torch.load('checkpoint/checkpoint.chk')
    predictor_module.load_state_dict(chk)

    test(test_set, predictor_module, loss_module)

if  __name__ == '__main__':
    run_test()
