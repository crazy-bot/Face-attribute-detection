
from data.test_dataset import MultitaskTestDataset
from model.network import MultitaskClassifier
import sys
import os
from time import time, ctime
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
import numpy as np
from utils.utils import IOStream, get_testargs
import warnings
warnings.filterwarnings("ignore")

class Tester:
    def __init__(self, dataroot, classification_type, batch_size, num_classes, ckpt_dir) -> None:
        
        self.dataroot = dataroot    #'/data/suparna/workspace/TinyPortraits_thumbnails/'
        
        #--------- loading dataset and creating loader
        ds = MultitaskTestDataset(dataroot, classification_type)
        self.le_gender = ds.le_gender
        self.le_haircolor = ds.le_haircolor
        self.test_loader = DataLoader(ds, batch_size=batch_size, num_workers=4) 
        print('test loader size', len(self.test_loader))

        #---------Initialize model, optimizer, loss function
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clf = MultitaskClassifier(device=self.device, num_classes=num_classes)
        self.clf.load_state_dict(torch.load(ckpt_dir))
        self.clf.eval()

    def test(self, io, output_file):
        time_str = 'Testing start:' + ctime(time())
        io.cprint(time_str)

        pred1list, pred2list = [], []
        label1list, label2list = [], []

        for i , (data, label1, label2) in enumerate(self.test_loader):
            data, label1, label2 = data.to(self.device), label1.to(self.device), label2.to(self.device)
            gender_pred, haircolor_pred = self.clf(data)
            #import pdb; pdb.set_trace()
            gender_pred = torch.argmax(gender_pred, dim=1)
            haircolor_pred = torch.argmax(haircolor_pred, dim=1)

            pred1list.extend(gender_pred.cpu().numpy())
            pred2list.extend(haircolor_pred.cpu().numpy())

            label1list.extend(label1.cpu().numpy())
            label2list.extend(label2.cpu().numpy())
        
        time_str = 'Testing end:' + ctime(time())
        io.cprint(time_str)
        
        report1 = classification_report(label1list, pred1list, target_names=self.le_gender)
        report2 = classification_report(label2list, pred2list, target_names=self.le_haircolor)
        
        pred1list = [self.le_gender[int(i)] for i in pred1list]
        pred2list = [self.le_haircolor[int(i)] for i in pred2list]
        pred1list = np.array(pred1list)
        pred2list = np.array(pred2list)
        
        np.savetxt(output_file.format('gender'), pred1list, fmt='%i')
        np.savetxt(output_file.format('haircolor'), pred2list, fmt='%i')
        log = 'classification_report of gender classification: \n {}'.format(report1)
        log += '\n classification_report of hair color classification: \n {} '.format(report2)
        io.cprint(log)
    


if __name__ == '__main__':
    try:
        args = get_testargs()

    except ValueError:
        print("Missing or invalid arguments")
        sys.exit(0)

    ckpt_dir = 'checkpoints/%s'%args.exp_name+'/models/%s'%args.ckpt+'_epoch.pt'
    assert os.path.exists(ckpt_dir),'model doest not exist: %s'%ckpt_dir

    io = IOStream('checkpoints/' + args.exp_name + '/test.log')
    io.cprint('Program start: %s' % ctime(time()))

    num_classes = {
        'gender': 2,
        'haircolor': 5
    }

    output_file = 'checkpoints/%s'%args.exp_name+'/predictions_{}.txt'
    tester = Tester(args.dataroot, args.classification_type, args.batch_size, num_classes, ckpt_dir)

    tester.test(io, output_file)


