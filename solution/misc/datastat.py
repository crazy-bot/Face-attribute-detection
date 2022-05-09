from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def train_test_division(labelfile, isStratify= False, strat_col = None):

    att_df = pd.read_csv(labelfile, header = 0, keep_default_na = False)
    #------- remove bad images
    bad_images = open('asset/bad_images.txt').read().split()
    bad_images = [int(i) for i in bad_images]
    for idx in bad_images:
        att_df.drop(att_df[att_df['Image_Index'] == idx].index,inplace=True)

    #--------- stratified split into train-test
    if isStratify:
        assert strat_col != None and 0 < strat_col < 4, 'strat_col must be specified within the 0-3'
        if strat_col == 2:
             #----------Drop unknown category
            att_df.drop(att_df[att_df['Hair_Color'] == 'n/a'].index,inplace=True)
        att_df = att_df.to_numpy()
        print (att_df.shape)

        X, y = att_df[:,0], att_df[:,strat_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)

        train_data = {
            'Image_Index': X_train,
            'labels': y_train
        }
        val_data = {
            'Image_Index': X_val,
            'labels': y_val
        }
        test_data = {
            'Image_Index': X_test,
            'labels': y_test
        }
        if strat_col == 1:
            filename = 'asset/gender_{}.csv'
        elif strat_col == 2:
            filename = 'asset/haircolor_{}.csv'
        saveToCSV(train_data, val_data, test_data, filename)

    else:
         #----------Drop unknown category
        att_df.drop(att_df[att_df['Hair_Color'] == 'n/a'].index,inplace=True)
        att_df = att_df.to_numpy()
        print (att_df.shape)
        
        X, y = att_df[:,0], att_df[:,1:3]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

        train_data = {
            'Image_Index': X_train,
            'labels_gender': y_train[:, 0],
            'labels_haircolor': y_train[:,1]
        }
        val_data = {
            'Image_Index': X_val,
            'labels_gender': y_val[:,0],
            'labels_haircolor': y_val[:,1]
        }
        test_data = {
            'Image_Index': X_test,
            'labels_gender': y_test[:,0],
            'labels_haircolor': y_test[:,1]
        }
        
        filename = 'asset/multitask_{}.csv'
        saveToCSV(train_data, val_data, test_data, filename)



def saveToCSV(train_data, val_data, test_data, filename):

    df = pd.DataFrame.from_dict(train_data)
    df.to_csv(filename.format('train'))

    df = pd.DataFrame.from_dict(val_data)
    df.to_csv(filename.format('val'))

    df = pd.DataFrame.from_dict(test_data)
    df.to_csv(filename.format('test'))

class MyDataset(Dataset):
    def __init__(self, dataroot, data):
        self.dataroot = dataroot
        self.data = data
        
    def __getitem__(self, index):
        i = self.data[index]
        frame_path = self.dataroot+'Tiny_Portrait_%06d.png'%int(i)
        x = Image.open(frame_path)
        x = torch.FloatTensor(np.array(x))
        x /= 255
        return x

    def __len__(self):
        return len(self.data)

    def calculate_norm_params(self, mydata):
            
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


if __name__ == '__main__':
    labelfile  = 'asset/Tiny_Portraits_Attributes.csv'
    train_test_division(labelfile, isStratify=True, strat_col=1)
    train_test_division(labelfile, isStratify=True, strat_col=2)
    train_test_division(labelfile)

    dataroot = '/data/suparna/workspace/TinyPortraits_thumbnails/'
    labelfile = 'asset/{}_train.csv'.format('multitask')
    train_df = pd.read_csv(labelfile, header = 0, keep_default_na = False)
    train_df = train_df.to_numpy()
    #import pdb;pdb.set_trace()
    X_train = train_df[:,1]
    mydata = MyDataset(dataroot, X_train)
    mean, std = mydata.calculate_norm_params(mydata)
    stat = {
        'mean': mean,
        'std': std
    }
    df = pd.DataFrame.from_dict(stat)
    df.to_csv('asset/datastat.csv')