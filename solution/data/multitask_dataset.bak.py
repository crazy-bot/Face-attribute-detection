import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import cv2
import torch 
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, dataroot, data):
        self.dataroot = dataroot
        self.data = data
        
    def __getitem__(self, index):
        i = self.data[index]
        frame_path = self.dataroot+'Tiny_Portrait_%06d.png'%i
        x = Image.open(frame_path)
        x = torch.FloatTensor(np.array(x))
        x /= 255
        return x

    def __len__(self):
        return len(self.data)

class DataSplitter:
    def __init__(self, dataroot, labelfile) -> None:

        self.dataroot = dataroot
        att_df = pd.read_csv(labelfile, header = 0, keep_default_na = False)
        att_df.loc[att_df['Hair_Color']=='n/a', 'Hair_Color'] = 'other'
        att_df.loc[att_df['Hair_Type']=='n/a', 'Hair_Type'] = 'other'
        
        #------- remove bad images
        bad_images = open('asset/bad_images.txt').read().split()
        bad_images = [int(i) for i in bad_images]
        for idx in bad_images:
            att_df.drop(att_df[att_df['Image_Index'] == idx].index,inplace=True)
            
        #image_index = att_df.index
        att_df = att_df.to_numpy()
        X, y = att_df[:,0], att_df[:,1:3]

        #--------Encode labels
        le_gender = preprocessing.LabelEncoder()
        att_df[:,1] = le_gender.fit_transform(att_df[:,1])
        self.le_gender = le_gender
        
        le_haircolor = preprocessing.LabelEncoder()
        att_df[:,2] = le_haircolor.fit_transform(att_df[:,2])
        self.le_haircolor = le_haircolor

        #--------- stratified split into train-val-test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

        # mu, sigma = self.__calculate_norm_params(dataroot, X_train)
        mu = torch.FloatTensor([0.5163, 0.4120, 0.3578])
        sigma = torch.FloatTensor([0.2682, 0.2379, 0.2292])

        #--------- weighted sampling
        self.train_sampler = self.__compute_sampler(y_train)
        self.train_dset = self.TinyFaceTrain(dataroot, X_train, y_train, mu, sigma, isaugment=True)
        self.val_dset = self.TinyFaceTrain(dataroot, X_val, y_val, mu, sigma, isaugment=False)
        self.X_test = X_test
        self.y_test = y_test

    def __calculate_norm_params(self, dataroot, X_train):
        
        
        mydata = MyDataset(dataroot, X_train)
        loader = DataLoader(
                mydata,
                batch_size=100,
                num_workers=1,
                shuffle=False
            )
        mean = 0.
        std = 0.
        nb_samples = 0.
        for data in loader:
            data = data.permute(0,3,1,2)
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples

        mean /= nb_samples
        std /= nb_samples
        print(mean, std) 
        # tensor([131.7485, 105.1266,  91.3141]) tensor([68.3852, 60.6654, 58.4380])
        # normalized -- tensor([0.5163, 0.4120, 0.3578]) tensor([0.2682, 0.2379, 0.2292])
        return mean, std

    def __compute_sampler(self, target):
        class_sample_count = np.unique(target, return_counts = True)[1]
        class_weight = 1/class_sample_count
        sample_weights = np.array([class_weight[t] for t in target])
        sample_weights = torch.from_numpy(sample_weights).double()
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        return sampler

    def get_loader(self, dset, batch_size, num_workers, sampler=None):
        loader = DataLoader(dset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
        return loader


    class TinyFaceTrain(Dataset):

        def __init__(self, dataroot, image_index, labels, mu, sigma, isaugment ):
            self.dataroot = dataroot
            self.data = image_index
            self.target = labels
            self.isaugment = isaugment
            if isaugment:
                transform = T.RandomApply(
                torch.nn.ModuleList([
                    T.CenterCrop(size=(80,80)),
                    T.RandomRotation(degrees=(-90, 90), resample = 0),
                    T.RandomPerspective(distortion_scale=0.4, p=1.0),
                    T.RandomHorizontalFlip(p=0.5)
                ]),p=0.3)
                self.rand_transform = torch.jit.script(transform)

            self.must_transform = torch.nn.Sequential(
                T.Resize(size=(80,80)),
                T.Normalize(mu, sigma))
            self.must_transform = torch.jit.script(self.must_transform)
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, index):
            i = self.data[index]
            frame_path = self.dataroot+'Tiny_Portrait_%06d.png'%i
            
            x = Image.open(frame_path) # width, height
            y = self.target[index]
            x = torch.FloatTensor(np.array(x)/255)
            y = torch.from_numpy(np.array(y))
            
            if self.isaugment: # check for train mode
                x = self.rand_transform(x)
            x = x.permute(2,0,1)
            x = self.must_transform(x)

            return x, y


if __name__ == '__main__':
    dataroot = '/data/suparna/workspace/TinyPortraits_thumbnails/'
    labelfile = 'solution/asset/Tiny_Portraits_Attributes.csv'
    ds  = DataSplitter(dataroot, labelfile, 1)

    train_loader = ds.get_loader(ds.train_dset, batch_size=2, num_workers=1, sampler=ds.train_sampler)

    val_loader = ds.get_loader(ds.val_dset, batch_size=10, num_workers=1)

    print('train loader size', len(train_loader))
    print('val loader size', len(val_loader))

    for data, label in train_loader:
        print(data.size(), label.size())
        print(torch.min(data), torch.max(data))
        break