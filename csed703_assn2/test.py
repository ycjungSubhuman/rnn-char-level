'''
Test/validation cycle definition
'''
import os
import numpy as np
from torch.utils.data.dataloader import DataLoader

import dataset
import vis
import model

def get_average_loss(predictor_module, loss_module, loader):
    "Get average loss of the whole test dataset"
    loss = 0.0
    cnt = 0
    for label, text in loader:
        pred = predictor_module(text)
        loss += loss_module(label, pred)
        cnt += 1
    return loss / cnt

def get_confusion_matrix(predictor_module, loader):
    "Get confusion matrix on the test dataset"
    num_class = len(loader.dataset.class_names)

    mat = np.zeros((num_class, num_class), dtype=np.int32)
    for data in loader:
        label, text = data
        pred = np.argmax(predictor_module(text).datach().numpy())
        mat[pred, label] += 1

    return mat

def test(test_set, predictor_module, loss_module,
         loss_hook=get_average_loss, 
         vis_hook=lambda predictor, loader:
         vis.save_confusion_matrix('confusion.png', loader.dataset.class_names,
                                   get_confusion_matrix(predictor, loader))):
    """
    Test model performance as accuracy

    This function can also be used for validation

    test_set        a dataset.FilenameClassDataset object
    model           loss calculation model
    loss_hook       a hook for calculating loss value.
                    takes 'model', dataloader  and returns loss
    vis_hook        a hook for visualizing test results
    """
    test_loader = DataLoader(test_set)

    loss = loss_hook(predictor_module, loss_module, test_loader)
    vis_hook(predictor_module, test_loader)

    return loss

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
    predictor_module.load_state_dict('checkpoint.chk')

    test(test_set, predictor_module, loss_module)

if  __name__ == '__main__':
    run_test()
