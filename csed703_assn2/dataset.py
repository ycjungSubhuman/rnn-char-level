'''
Dataset i/o glue codes and preprocessing
'''
import string
import unicodedata
import os
import codecs
import torch
from torch.utils.data.dataset import Dataset

class FilenameClassDataset(Dataset):
    """
    A dataset that has class label as filename,
    and data as lines of corresponding files
    """

    def __init__(self, root_path, transform=None):
        """
        root_path       the folder containing .txt files
        transform       transform to apply on text
        """
        self.transform = transform
        self.class_names = [
            os.path.splitext(f)[0]
            for f in os.listdir(root_path)
            if f.endswith('.txt')]

        def class_names_to_path(class_name):
            "given a class name, returns a full path"
            return os.path.join(root_path, class_name+'.txt')

        def path_to_lines(path):
            "given a path, returns string list of lines"
            with codecs.open(
                    path, "r",
                    encoding='utf-8',
                    errors='ignore') as f:
                lines = f.readlines()

            return [s.strip() for s in lines]

        self.lines_per_class_ind = [
            path_to_lines(class_names_to_path(cn)) 
            for cn in self.class_names]

    def __len__(self):
        return sum([len(li) for li in self.lines_per_class_ind])

    def class_ind_to_class_name(self, index):
        "Converts class index to class name"
        if index not in range(len(self.class_names)):
            raise ('Invalid index {}. Must range from 0 to {}'
                   .format(index, range(len(self.class_names))))
        return self.class_names[index]

    def __getitem__(self, idx):
        # Calculate in-class index (marginal_ind)
        class_ind = 0
        marginal_ind = idx
        while marginal_ind >= len(self.lines_per_class_ind[class_ind]):
            marginal_ind -= len(self.lines_per_class_ind[class_ind])
            class_ind += 1
        raw_line = self.lines_per_class_ind[class_ind][marginal_ind]
        line = (raw_line if self.transform is None
                else self.transform(raw_line))
        return class_ind, line

class TextlineToVector():
    "Text data to one-hot vector representation"

    def __init__(self, coding='ascii'):
        if coding == 'ascii':
            self.all_chars = string.ascii_letters + ".,;'`"
        else:
            raise 'Not Implemented'

    def latin_to_english(self, latin_string):
        """
        Converts Latin character to similar english characters"
        https://stackoverflow.com/a/518232/2809427
        """
        return ''.join(
            c for c in unicodedata.normalize('NFD', latin_string)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_chars)

    def ascii_char_to_index(self, char):
        "Convertx a ascii character to index"
        return self.all_chars.find(char)

    def __call__(self, data_in):
        line = self.latin_to_english(data_in)
        tensor = torch.zeros(len(line), 1, len(self.all_chars))
        for i, letter in enumerate(line):
            tensor[i, 0, self.ascii_char_to_index(letter)] = 1
        return tensor
