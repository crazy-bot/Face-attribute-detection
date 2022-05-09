import argparse

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

def get_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--dataroot', type=str, default='exp', metavar='N',
                        help='Path of the root dir of image dataset')

    parser.add_argument('--classification_type', type=str, default='gender', metavar='N',
                        choices = ['gender', 'haircolor', 'multitask'], help='which classification you want to train')
    
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
                        
    parser.add_argument('--batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
                        
    parser.add_argument('--epochs', type=int, metavar='N',
                        help='number of episode to train ')
                        
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    
    args = parser.parse_args()
    return args

def get_testargs():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--dataroot', type=str, default='exp', metavar='N',
                        help='Path of the root dir of image dataset')
                        
    parser.add_argument('--classification_type', type=str, default='gender', metavar='N',
                        choices = ['gender', 'haircolor', 'multitask'], help='which classification you want to train')
    
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    
    parser.add_argument('--batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
                        
    parser.add_argument('--ckpt', type=str, default='best', metavar='batch_size',
                        choices = ['best', 'latest'], help='Size of batch)')
                        
    args = parser.parse_args()
    return args