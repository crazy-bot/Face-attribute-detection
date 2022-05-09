import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import cv2
import torch 
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from PIL import Image



class DataSplitter:
    def __init__(self, dataroot, classification_type) -> None:

        self.dataroot = dataroot
        assert classification_type in ['multitask'], 'classification_type must be multitask'

        labelfile = 'asset/{}_train.csv'.format(classification_type)
        train_df = pd.read_csv(labelfile, header = 0, keep_default_na = False)
        train_df = train_df.to_numpy()
        X_train, y_train = train_df[:,1], train_df[:,2:]

        labelfile = 'asset/{}_val.csv'.format(classification_type)
        val_df = pd.read_csv(labelfile, header = 0, keep_default_na = False)
        val_df = val_df.to_numpy()
        X_val, y_val = val_df[:,1], val_df[:,2:]

         #--------Encode labels
        le_gender = preprocessing.LabelEncoder()
        y_train[:,0] = le_gender.fit_transform(y_train[:,0])
        y_val[:,0] = le_gender.transform(y_val[:,0])
        self.le_gender = le_gender
        
        le_haircolor = preprocessing.LabelEncoder()
        y_train[:,1] = le_haircolor.fit_transform(y_train[:,1])
        y_val[:,1] = le_haircolor.transform(y_val[:,1])
        self.le_haircolor = le_haircolor

        #------calculate class weights
        
        class_weights = []
        for i in range(y_train.shape[1]):
            class_weights.append(self.calculate_weights(y_train[:,i]))
        self.class_weights = np.array(class_weights)
        
        #import pdb; pdb.set_trace()
        #y = y.reshape(-1,1)
        #oe = preprocessing.OneHotEncoder()
        #y = oe.fit_transform(y).toarray()

        mu = torch.FloatTensor([0.5163, 0.4120, 0.3578])
        sigma = torch.FloatTensor([0.2682, 0.2379, 0.2292])

        self.train_dset = self.TinyFaceTrain(dataroot, X_train, y_train, mu, sigma, isaugment=True)
        self.val_dset = self.TinyFaceTrain(dataroot, X_val, y_val, mu, sigma, isaugment=False)
          
    # def __compute_sampler(self, target):
    #     class_sample_count = np.unique(target, return_counts = True)[1]
    #     class_weight = 1/class_sample_count
    #     sample_weights = np.array([class_weight[t] for t in target])
    #     sample_weights = torch.from_numpy(sample_weights).double()
    #     sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    #     return sampler

    def get_loader(self, dset, batch_size, num_workers, sampler=None):
        loader = DataLoader(dset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
        return loader
    
    def calculate_weights(self, labels):
        unique, counts = np.unique(labels, return_counts = True)
        class_weights = counts / np.sum(counts)
        class_weights = class_weights.astype(np.float32)
        return class_weights
        

    class TinyFaceTrain(Dataset):

        def __init__(self, dataroot, image_index, labels, mu, sigma, isaugment ):
            self.dataroot = dataroot
            self.data = image_index
            self.target = labels
            self.isaugment = isaugment
            if isaugment:
                transform = T.RandomApply(
                torch.nn.ModuleList([
                    #T.CenterCrop(size=(80,80)),
                    T.RandomRotation(degrees=(-90, 90), resample = 0),
                    T.RandomPerspective(distortion_scale=0.4, p=1.0),
                    T.RandomHorizontalFlip(p=0.5)
                ]),p=0.3)
                self.rand_transform = torch.jit.script(transform)

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
            
            if self.isaugment: # check for train mode
                x = self.rand_transform(x)
            x = x.permute(2,0,1)
            x = self.must_transform(x)

            return x, label1, label2


if __name__ == '__main__':
    dataroot = '/data/suparna/workspace/TinyPortraits_thumbnails/'
    labelfile = 'solution/asset/Tiny_Portraits_Attributes.csv'
    ds  = DataSplitter(dataroot, 'multitask')

    train_loader = ds.get_loader(ds.train_dset, batch_size=2, num_workers=1)

    val_loader = ds.get_loader(ds.val_dset, batch_size=10, num_workers=1)

    print('train loader size', len(train_loader))
    print('val loader size', len(val_loader))

    for data, label in train_loader:
        print(data.size(), label.size())
        print(torch.min(data), torch.max(data))
        break