from ast import Num
import sys
from turtle import forward
sys.path.append('/data/suparna/workspace/')
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn as nn
import torch

class FaceClassifier(nn.Module):
    def __init__(self, device, num_classes) -> None:
        super().__init__()
        
        resnet = InceptionResnetV1(
        classify=True,
        pretrained='vggface2',
        num_classes = num_classes
        ).to(device)

        self.model = resnet
        

    def forward(self, x):
        x = self.model(x)
        return x

class MultitaskClassifier(nn.Module):
    def __init__(self, device, num_classes) -> None:
        super().__init__()
        
        resnet = InceptionResnetV1(
        classify=True,
        pretrained='vggface2'
        ).to(device)

        self.feat_layers = list(resnet.children())
        self.feat_layers = self.feat_layers[:-3]
        self.feat_layers = nn.Sequential(*self.feat_layers).to(device)

        embedding_dim = 1792
        out_dim = list(num_classes.values())
        self.expert1 = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, out_dim[0]),
        ).to(device)
        self.expert2 = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, out_dim[1]),
        ).to(device)
        

    def forward(self, x):
        batch_size = x.size(0)
        x_feat = self.feat_layers(x)
        x_feat = x_feat.view(batch_size, -1)
        out1 = self.expert1(x_feat)
        out2 = self.expert2(x_feat)
        return out1, out2

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clf = MultitaskClassifier(device=device, num_classes=2)
