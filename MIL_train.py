import sys
import os
import numpy as np
import argparse
import random
import openslide
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models

parser = argparse.ArgumentParser(description='MIL-nature-medicine-2019 tile classifier training script')
parser.add_argument('--train_lib', type=str, default='', help='path to train MIL library binary')
parser.add_argument('--val_lib', type=str, default='', help='path to validation MIL library binary. If present.')
parser.add_argument('--output', type=str, default='.', help='name of output file')
parser.add_argument('--batch_size', type=int, default=512, help='mini-batch size (default: 512)')
parser.add_argument('--nepochs', type=int, default=100, help='number of epochs')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--test_every', default=10, type=int, help='test on val every (default: 10)')
parser.add_argument('--weights', default=0.5, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
parser.add_argument('--k', default=1, type=int, help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')

best_acc = 0
def main():
    global args, best_acc
    args = parser.parse_args()

    #cnn
    model = models.resnet34(True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.cuda()

    if args.weights==0.5:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        w = torch.Tensor([1-args.weights,args.weights])
        criterion = nn.CrossEntropyLoss(w).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    cudnn.benchmark = True

    #normalization
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    #load data
    train_dset = MILdataset(args.train_lib, trans)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    if args.val_lib:
        val_dset = MILdataset(args.val_lib, trans)
        val_loader = torch.utils.data.DataLoader(
            val_dset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False)

    #open output file
    fconv = open(os.path.join(args.output,'convergence.csv'), 'w')
    fconv.write('epoch,metric,value\n')
    fconv.close()

    #loop throuh epochs
    for epoch in range(args.nepochs):
        train_dset.setmode(1)
        probs = inference(epoch, train_loader, model)
        topk = group_argtopk(np.array(train_dset.slideIDX), probs, args.k)
        train_dset.maketraindata(topk)
        train_dset.shuffletraindata()
        train_dset.setmode(2)
        loss = train(epoch, train_loader, model, criterion, optimizer)
        print('Training\tEpoch: [{}/{}]\tLoss: {}'.format(epoch+1, args.nepochs, loss))
        fconv = open(os.path.join(args.output, 'convergence.csv'), 'a')
        fconv.write('{},loss,{}\n'.format(epoch+1,loss))
        fconv.close()

        #Validation
        if args.val_lib and (epoch+1) % args.test_every == 0:
            val_dset.setmode(1)
            probs = inference(epoch, val_loader, model)
            maxs = group_max(np.array(val_dset.slideIDX), probs, len(val_dset.targets))
            pred = [1 if x >= 0.5 else 0 for x in maxs]
            err,fpr,fnr = calc_err(pred, val_dset.targets)
            print('Validation\tEpoch: [{}/{}]\tError: {}\tFPR: {}\tFNR: {}'.format(epoch+1, args.nepochs, err, fpr, fnr))
            fconv = open(os.path.join(args.output, 'convergence.csv'), 'a')
            fconv.write('{},error,{}\n'.format(epoch+1, err))
            fconv.write('{},fpr,{}\n'.format(epoch+1, fpr))
            fconv.write('{},fnr,{}\n'.format(epoch+1, fnr))
            fconv.close()
            #Save best model
            err = (fpr+fnr)/2.
            if 1-err >= best_acc:
                best_acc = 1-err
                obj = {
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict()
                }
                torch.save(obj, os.path.join(args.output,'checkpoint_best.pth'))

def inference(run, loader, model):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i, input in enumerate(loader):
            print('Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(run+1, args.nepochs, i+1, len(loader)))
            input = input.cuda()
            output = F.softmax(model(input), dim=1)
            probs[i*args.batch_size:i*args.batch_size+input.size(0)] = output.detach()[:,1].clone()
    return probs.cpu().numpy()

def train(run, loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.
    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*input.size(0)
    return running_loss/len(loader.dataset)

def calc_err(pred,real):
    pred = np.array(pred)
    real = np.array(real)
    neq = np.not_equal(pred, real)
    err = float(neq.sum())/pred.shape[0]
    fpr = float(np.logical_and(pred==1,neq).sum())/(real==0).sum()
    fnr = float(np.logical_and(pred==0,neq).sum())/(real==1).sum()
    return err, fpr, fnr

def group_argtopk(groups, data,k=1):
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-k:] = True
    index[:-k] = groups[k:] != groups[:-k]
    return list(order[index])

def group_max(groups, data, nmax):
    out = np.empty(nmax)
    out[:] = np.nan
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    out[groups[index]] = data[index]
    return out

class MILdataset(data.Dataset):
    def __init__(self, libraryfile='', transform=None):
        lib = torch.load(libraryfile)
        slides = []
        for i,name in enumerate(lib['slides']):
            sys.stdout.write('Opening SVS headers: [{}/{}]\r'.format(i+1, len(lib['slides'])))
            sys.stdout.flush()
            slides.append(openslide.OpenSlide(name))
        print('')
        #Flatten grid
        grid = []
        slideIDX = []
        for i,g in enumerate(lib['grid']):
            grid.extend(g)
            slideIDX.extend([i]*len(g))

        print('Number of tiles: {}'.format(len(grid)))
        self.slidenames = lib['slides']
        self.slides = slides
        self.targets = lib['targets']
        self.grid = grid
        self.slideIDX = slideIDX
        self.transform = transform
        self.mode = None
        self.mult = lib['mult']
        self.size = int(np.round(224*lib['mult']))
        self.level = lib['level']
    def setmode(self,mode):
        self.mode = mode
    def maketraindata(self, idxs):
        self.t_data = [(self.slideIDX[x],self.grid[x],self.targets[self.slideIDX[x]]) for x in idxs]
    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))
    def __getitem__(self,index):
        if self.mode == 1:
            slideIDX = self.slideIDX[index]
            coord = self.grid[index]
            img = self.slides[slideIDX].read_region(coord,self.level,(self.size,self.size)).convert('RGB')
            if self.mult != 1:
                img = img.resize((224,224),Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img
        elif self.mode == 2:
            slideIDX, coord, target = self.t_data[index]
            img = self.slides[slideIDX].read_region(coord,self.level,(self.size,self.size)).convert('RGB')
            if self.mult != 1:
                img = img.resize((224,224),Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img, target
    def __len__(self):
        if self.mode == 1:
            return len(self.grid)
        elif self.mode == 2:
            return len(self.t_data)

if __name__ == '__main__':
    main()
