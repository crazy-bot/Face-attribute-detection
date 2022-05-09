
from data.test_dataset import TinyFaceTestDataset
from model.network import FaceClassifier, MultitaskClassifier
import sys
import os
from time import time, ctime
import torch
import pandas as pd
from utils.utils import IOStream, get_testargs
import torchvision.transforms as T
import numpy as np
from PIL import Image
import csv

import warnings
warnings.filterwarnings("ignore")

class Tester:
    def __init__(self, dataroot, classification_type, num_classes, ckpt_dir) -> None:
        
        self.dataroot = dataroot

        #--------- loading dataset and creating loader
        # ds = TinyFaceTestDataset(dataroot, classification_type)
        # self.target_names = ds.target_names
        # self.target_names = np.array(self.target_names)
        # np.save('asset/haircolor_mapping.npy', self.target_names)

        if classification_type == 'gender':
            self.target_names = np.load('asset/gender_mapping.npy', allow_pickle=True)
        if classification_type == 'haircolor':
            self.target_names = np.load('asset/haircolor_mapping.npy', allow_pickle=True)
        if classification_type == 'multitask':
            self.le_gender = np.load('asset/gender_mapping.npy', allow_pickle=True)
            self.le_haircolor = np.load('asset/haircolor_mapping.npy', allow_pickle=True)

        #---------Initialize model, optimizer, loss function
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if classification_type in ['gender', 'haircolor']:
            self.clf = FaceClassifier(device=self.device, num_classes=num_classes)
        elif classification_type == 'multitask':
            self.clf = MultitaskClassifier(device=self.device, num_classes=num_classes)
        self.clf.load_state_dict(torch.load(ckpt_dir))
        self.clf.eval()

        #--------Initialize transform
        stat_df = pd.read_csv('asset/datastat.csv', header = 0, keep_default_na = False)
        stat_df = stat_df.to_numpy()
        stat_df = stat_df.T[1]
        mu, sigma = stat_df[1], stat_df[2]
        self.must_transform = T.Compose([
                T.Resize(size=(80,80)),
                T.Normalize(mu, sigma)])

    def test(self, io, output_file):
        time_str = 'Prediction start:' + ctime(time())
        io.cprint(time_str)

        header = ['filename', 'prediction']
        file = open(output_file, 'w')
        writer = csv.writer(file)
        writer.writerow(header)

        for f in os.listdir(self.dataroot):
            frame_path = os.path.join(self.dataroot, f)
            x = Image.open(frame_path) #width, height
            x = x.convert('RGB') 
            x = torch.FloatTensor(np.array(x)/255)
            x = x.permute(2,0,1)
            x = self.must_transform(x)
            x = x.unsqueeze(0)
            x = x.to(self.device)
            #import pdb; pdb.set_trace()
            pred = self.clf(x)
            pred = torch.argmax(pred, dim=1)
            pred = pred[0].cpu().numpy()
            pred = self.target_names[pred]
            
            log = 'prediction of %s %s'%(frame_path, pred)
            io.cprint(log)
            writer.writerow([f, pred])
        
        time_str = 'Testing end:' + ctime(time())
        io.cprint(time_str)
        file.close()
        

    def test_multitask(self, io, output_file):
        time_str = 'Testing start:' + ctime(time())
        io.cprint(time_str)

        header = ['filename', 'prediction']
        file = open(output_file, 'w')
        writer = csv.writer(file)
        writer.writerow(header)

        for f in os.listdir(self.dataroot):
            frame_path = os.path.join(self.dataroot, f)
            x = Image.open(frame_path) #width, height
            x = x.convert('RGB') 
            x = torch.FloatTensor(np.array(x)/255)
            x = x.permute(2,0,1)
            x = self.must_transform(x)
            x = x.unsqueeze(0)
            x = x.to(self.device)
            gender_pred, haircolor_pred = self.clf(x)
            gender_pred = torch.argmax(gender_pred, dim=1)
            haircolor_pred = torch.argmax(haircolor_pred, dim=1)
        
            gender_pred = gender_pred[0].cpu().numpy()
            haircolor_pred = haircolor_pred[0].cpu().numpy()
            
            gender_pred = self.le_gender[gender_pred]
            haircolor_pred = self.le_haircolor[haircolor_pred]
            pred = gender_pred + ' '+ haircolor_pred
            log = 'prediction of %s %s'%(frame_path, pred)
            io.cprint(log)
            writer.writerow([f, pred])
        
        time_str = 'Testing end:' + ctime(time())
        io.cprint(time_str)
        file.close()

if __name__ == '__main__':
    try:
        args = get_testargs()

    except ValueError:
        print("Missing or invalid arguments")
        sys.exit(0)

    ckpt_dir = 'checkpoints/%s'%args.exp_name+'/models/%s'%args.ckpt+'_epoch.pt'
    assert os.path.exists(ckpt_dir),'model doest not exist: %s'%ckpt_dir

    io = IOStream('checkpoints/' + args.exp_name + '/pred.log')
    io.cprint('Program start: %s' % ctime(time()))

    if args.classification_type == 'gender':
        output_file = 'checkpoints/%s'%args.exp_name+'/predictions_gender.csv'
        num_classes = 2
    if args.classification_type == 'haircolor':
        output_file = 'checkpoints/%s'%args.exp_name+'/predictions_haircolor.csv'
        num_classes = 5
    if args.classification_type == 'multitask':
        output_file = 'checkpoints/%s'%args.exp_name+'/predictions_multitask.csv'
        num_classes = {
        'gender': 2,
        'haircolor': 5
        }
    
    tester = Tester(args.dataroot, args.classification_type, num_classes, ckpt_dir)
    if args.classification_type in ['gender', 'haircolor']:
        tester.test(io, output_file)
    elif args.classification_type == 'multitask':
        tester.test_multitask(io, output_file)
