from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split 
        self.FileInput = os.getcwd()+"/Caltech101/"+split+".txt"
        self.Img_list=[]
        self.Label_list=[]  
        self.Lenght=0

        file1 = open(self.FileInput, 'r')
        Lines = file1.readlines()
        
        n_label=0
        label_dict={}

        for line in Lines:
          label = line.strip().split("/")[0]
          if label=="BACKGROUND_Google":
            continue
          if label_dict.get(label)==None:
            label_dict[label]=n_label
            n_label+=1


          self.Img_list.append(pil_loader(root+"/"+line.strip()))
          self.Label_list.append(label_dict[label])
          self.Lenght += 1

        print("Loaded {} Images from {}".format(self.Lenght,split+".txt"))
        print(label_dict)
        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image = self.Img_list[index]
        label = self.Label_list[index]


        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
      return self.Lenght
