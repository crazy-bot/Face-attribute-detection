
from data.test_dataset import TinyFaceTestDataset
from model.network import FaceClassifier
import sys
import os
from time import time, ctime
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import recall_score, precision_score, accuracy_score
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
        ds = TinyFaceTestDataset(dataroot, classification_type)
        self.target_names = ds.target_names
        self.test_loader = DataLoader(ds, batch_size=batch_size, num_workers=4)        
        print('test loader size', len(self.test_loader))

        #---------Initialize model, optimizer, loss function
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clf = FaceClassifier(device=self.device, num_classes=num_classes)
        self.clf.load_state_dict(torch.load(ckpt_dir))
        self.clf.eval()

    def test(self, io, output_file):
        time_str = 'Testing start:' + ctime(time())
        io.cprint(time_str)

        predlist = []
        labellist = []
        for i , (data, label, ) in enumerate(self.test_loader):
            data, label = data.to(self.device), label.to(self.device)
            import pdb; pdb.set_trace()
            pred = self.clf(data)
            pred = torch.argmax(pred, dim=1)
            predlist.extend(pred.cpu().numpy())
            labellist.extend(label.cpu().numpy())
        
        time_str = 'Testing end:' + ctime(time())
        io.cprint(time_str)
        
        report = classification_report(labellist, predlist, target_names=self.target_names)
        predlist = [self.target_names[int(i)] for i in predlist]
        predlist = np.array(predlist)
        np.savetxt(output_file, predlist, fmt='%s')
        log = 'classification_report: \n {} '.format(report)
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

    if args.classification_type == 'gender':
        column_num = 1
        num_classes = 2
    if args.classification_type == 'haircolor':
        column_num = 2
        num_classes = 5

    output_file = 'checkpoints/%s'%args.exp_name+'/predictions.txt'
    tester = Tester(args.dataroot, args.classification_type, args.batch_size, num_classes, ckpt_dir)

    tester.test(io, output_file)


