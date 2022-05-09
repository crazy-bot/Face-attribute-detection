from data.multitask_dataset import DataSplitter
from model.network import MultitaskClassifier
import sys
import os
from time import time, ctime
import torch
import torch.nn.functional as F
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
        self.le_gender = ds.le_gender.classes_
        self.le_haircolor = ds.le_haircolor.classes_

        self.train_loader = ds.get_loader(ds.train_dset, batch_size=batch_size, num_workers=4)
        self.val_loader = ds.get_loader(ds.val_dset, batch_size=batch_size, num_workers=4)
        print('train loader size', len(self.train_loader))
        print('val loader size', len(self.val_loader))

        #---------Initialize model, optimizer, loss function
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clf = MultitaskClassifier(device=self.device, num_classes=num_classes)
        #self.clf = torch.nn.DataParallel(self.clf)
        class_weights = []
        class_weights.append(torch.FloatTensor(ds.class_weights[0]).to(self.device))
        class_weights.append(torch.FloatTensor(ds.class_weights[1]).to(self.device))
        self.class_weights = class_weights

        #optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, clf.parameters()), lr=0.001)
        self.optimizer = torch.optim.Adam(self.clf.parameters(), lr=0.001, weight_decay=1e-4)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def __training_loop(self, epoch, io):
        time_str = 'Train start:' + ctime(time())
        io.cprint(time_str)

        self.clf.train()
        train_loss = 0
        pred1list, pred2list = [], []
        label1list, label2list = [], []

        for i , (data, label1, label2) in enumerate(self.train_loader):
            data, label1, label2 = data.to(self.device), label1.to(self.device), label2.to(self.device)
            gender_pred, haircolor_pred = self.clf(data)
            
            gender_loss = F.cross_entropy(gender_pred, label1, weight=self.class_weights[0], reduction='mean')
            haircolor_loss = F.cross_entropy(haircolor_pred, label2, weight=self.class_weights[1], reduction='mean')
            loss = gender_loss + haircolor_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            
            self.optimizer.step()
            train_loss += loss.item()
            gender_pred = torch.argmax(gender_pred, dim=1)
            haircolor_pred = torch.argmax(haircolor_pred, dim=1)

            pred1list.extend(gender_pred.cpu().numpy())
            pred2list.extend(haircolor_pred.cpu().numpy())

            label1list.extend(label1.cpu().numpy())
            label2list.extend(label2.cpu().numpy())

            if i%10 == 0:
                log = 'Training epoch:{} iteration: {} gender_loss:{} haircolor_loss:{}'.format(epoch, i, gender_loss, haircolor_loss)
                io.cprint(log)

        time_str = 'Train end:' + ctime(time())
        io.cprint(time_str)

        train_loss /= len(self.train_loader)
        report1 = classification_report(label1list, pred1list, target_names=self.le_gender)
        report2 = classification_report(label2list, pred2list, target_names=self.le_haircolor)

        pack = {
            'report1': report1,
            'report2': report2,
            'loss': train_loss
        }
        return pack

    def __validation_loop(self, epoch, io):
        time_str = 'Validation start:' + ctime(time())
        io.cprint(time_str)

        self.clf.eval()
        val_loss = 0
        pred1list, pred2list = [], []
        label1list, label2list = [], []

        for i , (data, label1, label2) in enumerate(self.train_loader):
            data, label1, label2 = data.to(self.device), label1.to(self.device), label2.to(self.device)
            gender_pred, haircolor_pred = self.clf(data)
            #import pdb; pdb.set_trace()
            gender_loss = F.cross_entropy(gender_pred, label1, weight=self.class_weights[0], reduction='mean')
            haircolor_loss = F.cross_entropy(haircolor_pred, label2, weight=self.class_weights[1], reduction='mean')
            loss = 0.4*gender_loss + 0.6*haircolor_loss

            val_loss += loss.item()
            gender_pred = torch.argmax(gender_pred, dim=1)
            haircolor_pred = torch.argmax(haircolor_pred, dim=1)

            pred1list.extend(gender_pred.cpu().numpy())
            pred2list.extend(haircolor_pred.cpu().numpy())

            label1list.extend(label1.cpu().numpy())
            label2list.extend(label2.cpu().numpy())

            if i%10 == 0:
                log = 'Training epoch:{} iteration: {} gender_loss:{} haircolor_loss:{}'.format(epoch, i, gender_loss, haircolor_loss)
                io.cprint(log)
        
        time_str = 'Validation end:' + ctime(time())
        io.cprint(time_str)

        val_loss /= len(self.val_loader)
        
        report1 = classification_report(label1list, pred1list, target_names=self.le_gender)
        report2 = classification_report(label2list, pred2list, target_names=self.le_haircolor)
        prec1 = precision_score(label1list, pred1list, average = 'micro')
        prec2 = precision_score(label2list, pred2list, average = 'micro')
        pack = {
            'prec1': prec1,
            'prec2': prec2,
            'report1': report1,
            'report2': report2,
            'loss': val_loss
        }
        return pack

    def train(self, io):
        best_prec = 0.0
        for epoch in range(self.epochs):
            io.cprint('---------------------Epoch %d/%d---------------------' % (epoch, args.epochs))
            
            pack1 = self.__training_loop(epoch, io)
            self.writer.add_scalar('train/loss', pack1['loss'], epoch)
            log = 'Training epoch:{} loss:{} '.format(epoch, pack1['loss'])
            log += '\n %s'%pack1['report1'] + '\n %s'%pack1['report2']
            io.cprint(log)

            pack2 = self.__validation_loop(epoch, io)
            self.writer.add_scalar('valid/loss', pack2['loss'], epoch)
            log = 'Validation epoch:{} loss:{}'.format(epoch, pack2['loss'])
            log += '\n %s'%pack2['report1'] + '\n %s'%pack2['report2']
            io.cprint(log)

            precision = pack2['prec1'] + pack2['prec2']
            #----------save best valid precision model
            if precision > best_prec:
                torch.save(self.clf.state_dict(), '{}/{}.pt'.format(ckpt_dir,'best_epoch'))
                best_prec = precision

            if epoch == self.epochs-1:
                torch.save(self.clf.state_dict(), '{}/{}.pt'.format(ckpt_dir,'latest_epoch'))

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

    if args.classification_type != 'multitask':
        print("Missing or invalid arguments")
        sys.exit(0)

    num_classes = {
        'gender': 2,
        'haircolor': 5
    }

    #------ logging the hyperparameters settings
    for arg in vars(args):
        io.cprint('{} : {}'.format(arg, getattr(args, arg) ))

    trainer = Trainer(args.epochs, args.batch_size, args.dataroot, 
                args.classification_type, args.lr, num_classes)

    trainer.train(io)


