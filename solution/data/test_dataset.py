from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np
import data.tinyPortrait_dataset as tp
import data.multitask_dataset as mt

class TinyFaceTestDataset(Dataset):

    def __init__(self, dataroot, classification_type):

        assert classification_type in ['gender', 'haircolor'], 'classification_type must be gender or haircolor'
        self.dataroot = dataroot
        ds = tp.DataSplitter(dataroot, classification_type)
        self.target_names = ds.le.classes_

        labelfile = 'asset/{}_test.csv'.format(classification_type)
        test_df = pd.read_csv(labelfile, header = 0, keep_default_na = False)
        test_df = test_df.to_numpy()
        X_test, y_test = test_df[:,1], test_df[:,2]
        self.data = X_test
        y_test = ds.le.transform(y_test)
        self.label = y_test
        
        mu = torch.FloatTensor([0.5163, 0.4120, 0.3578])
        sigma = torch.FloatTensor([0.2682, 0.2379, 0.2292])

        self.must_transform = T.Compose([
                T.Resize(size=(80,80)),
                T.Normalize(mu, sigma)])
  
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        i = self.data[index]
        frame_path = self.dataroot+'Tiny_Portrait_%06d.png'%i
        x = Image.open(frame_path) #width, height
        y = self.label[index]
        x = torch.FloatTensor(np.array(x)/255)
        y = torch.from_numpy(np.array(y))
            
        x = x.permute(2,0,1)
        x = self.must_transform(x)

        return x, y

class MultitaskTestDataset(Dataset):

    def __init__(self, dataroot, classification_type):

        assert classification_type == 'multitask','classification_type must be multitask'

        self.dataroot = dataroot
        ds = mt.DataSplitter(dataroot, classification_type)
        self.le_gender = ds.le_gender.classes_
        self.le_haircolor = ds.le_haircolor.classes_

        labelfile = 'asset/{}_test.csv'.format(classification_type)
        test_df = pd.read_csv(labelfile, header = 0, keep_default_na = False)
        test_df = test_df.to_numpy()
        X_test, y_test = test_df[:,1], test_df[:,2:]
        self.data = X_test

       #--------Encode labels
        y_test[:,0] = ds.le_gender.transform(y_test[:,0])
        y_test[:,1] = ds.le_haircolor.transform(y_test[:,1])
        self.target = y_test
        
        mu = torch.FloatTensor([0.5163, 0.4120, 0.3578])
        sigma = torch.FloatTensor([0.2682, 0.2379, 0.2292])

        self.must_transform = T.Compose([
                T.Resize(size=(80,80)),
                T.Normalize(mu, sigma)])
  
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
            i = self.data[index]
            frame_path = self.dataroot+'Tiny_Portrait_%06d.png'%i

            x = Image.open(frame_path) #width, height
            y = self.target[index]
            x = torch.FloatTensor(np.array(x)/255)
            label1 = torch.from_numpy(np.array(y[0]))
            label2 = torch.from_numpy(np.array(y[1]))
        
            x = x.permute(2,0,1)
            x = self.must_transform(x)

            return x, label1, label2
