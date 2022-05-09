
from cv2 import preCornerDetect
from numpy import average
from data.tinyPortrait_dataset import DataSplitter
from model.network import FaceClassifier
import sys
import os
from time import time, ctime
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.metrics import classification_report
from utils.utils import IOStream, get_args

import warnings
warnings.filterwarnings("ignore")

class Trainer:
    def __init__(self, epochs, batch_size, dataroot, classification_type, lr, num_classes) -> None:
        #--------- set hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.writer = SummaryWriter()
        self.lr = lr
        self.dataroot = dataroot    #'/data/suparna/workspace/TinyPortraits_thumbnails/'
        
        #--------- loading dataset and creating loader
        ds = DataSplitter(dataroot, classification_type)
        self.target_names = ds.le.classes_
        self.train_loader = ds.get_loader(ds.train_dset, batch_size=batch_size, num_workers=4, sampler=ds.train_sampler)
        self.val_loader = ds.get_loader(ds.val_dset, batch_size=batch_size, num_workers=4)
        print('train loader size', len(self.train_loader))
        print('val loader size', len(self.val_loader))

        #---------Initialize model, optimizer, loss function
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clf = FaceClassifier(device=self.device, num_classes=num_classes)
        #self.clf = torch.nn.DataParallel(self.clf)
        print("Let's use", torch.cuda.device_count(), "GPUs!")

        #optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, clf.parameters()), lr=0.001)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.clf.parameters()), lr=0.001)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def __training_loop(self, epoch, io):
        time_str = 'Train start:' + ctime(time())
        io.cprint(time_str)

        self.clf.train()
        train_loss = 0
        predlist = []
        labellist = []
        for i , (data, label) in enumerate(self.train_loader):
            data, label = data.to(self.device), label.to(self.device)
            #import pdb; pdb.set_trace()
            pred = self.clf(data)
            loss = self.loss_fn(pred, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            pred = torch.argmax(pred, dim=1)
            predlist.extend(pred.cpu().numpy())
            labellist.extend(label.cpu().numpy())

            if i%10 == 0:
                log = 'Training epoch:{} iteration: {} loss:{}'.format(epoch, i, loss)
                io.cprint(log)

        time_str = 'Train end:' + ctime(time())
        io.cprint(time_str)

        train_loss /= len(self.train_loader)
        report = classification_report(labellist, predlist, target_names=self.target_names)
        prec = precision_score(labellist, predlist, average = 'micro')
        recall = recall_score(labellist, predlist, average = 'micro')
        pack = {
            'prec': prec,
            'recall': recall,
            'report': report,
            'loss': train_loss
        }
        return pack

    def __validation_loop(self, epoch, io):
        time_str = 'Validation start:' + ctime(time())
        io.cprint(time_str)

        self.clf.eval()
        val_loss = 0
        predlist = []
        labellist = []
        for i , (data, label) in enumerate(self.val_loader):
            data, label = data.to(self.device), label.to(self.device)
            
            pred = self.clf(data)
            loss = self.loss_fn(pred, label)
            val_loss += loss.item()
            pred = torch.argmax(pred, dim=1)
            predlist.extend(pred.cpu().numpy())
            labellist.extend(label.cpu().numpy())

            if i%10 == 0:
                log = 'Validation epoch:{} iteration: {} loss:{}'.format(epoch, i, loss)
                io.cprint(log)
        
        time_str = 'Validation end:' + ctime(time())
        io.cprint(time_str)

        val_loss /= len(self.val_loader)
        
        report = classification_report(labellist, predlist, target_names=self.target_names)
        prec = precision_score(labellist, predlist, average = 'micro')
        recall = recall_score(labellist, predlist, average = 'micro')
        pack = {
            'prec': prec,
            'recall': recall,
            'report': report,
            'loss': val_loss
        }
        return pack

    def train(self, io):
        best_prec = 0.0
        for epoch in range(self.epochs):
            io.cprint('---------------------Epoch %d/%d---------------------' % (epoch, args.epochs))
            
            pack = self.__training_loop(epoch, io)
            self.writer.add_scalar('train/loss', pack['loss'], epoch)
            self.writer.add_scalar('train/precision', pack['prec'], epoch)
            self.writer.add_scalar('train/recall', pack['recall'], epoch)
            log = 'Training epoch:{} loss:{} '.format(epoch, pack['loss'])
            log += '\n %s'%pack['report']
            io.cprint(log)

            pack = self.__validation_loop(epoch, io)
            self.writer.add_scalar('valid/loss', pack['loss'], epoch)
            self.writer.add_scalar('valid/precision', pack['prec'], epoch)
            self.writer.add_scalar('valid/recall', pack['recall'], epoch)
            log = 'Validation epoch:{} loss:{}'.format(epoch, pack['loss'])
            log += '\n %s'%pack['report']
            io.cprint(log)


            #----------save best valid precision model
            if pack['prec'] > best_prec:
                torch.save(self.clf.state_dict(), '{}/{}.pt'.format(ckpt_dir,'best_epoch'))
                best_prec = pack['prec']

            if epoch == self.epochs-1:
                torch.save(self.clf.state_dict(), '{}}/{}.pt'.format(ckpt_dir,'latest_epoch'))

        self.writer.close()



if __name__ == '__main__':
    try:
        args = get_args()

    except ValueError:
        print("Missing or invalid arguments")
        sys.exit(0)

    ckpt_dir = 'checkpoints/%s'%args.exp_name+'/models'
    os.makedirs(ckpt_dir, exist_ok = True)

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint('Program start: %s' % ctime(time()))

    if args.classification_type == 'gender':
        column_num = 1
        num_classes = 2
    if args.classification_type == 'haircolor':
        column_num = 2
        num_classes = 5

    #------ logging the hyperparameters settings
    for arg in vars(args):
        io.cprint('{} : {}'.format(arg, getattr(args, arg) ))

    trainer = Trainer(args.epochs, args.batch_size, args.dataroot, 
                args.classification_type, args.lr, num_classes)

    trainer.train(io)


