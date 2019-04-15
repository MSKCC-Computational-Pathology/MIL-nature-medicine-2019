
parser = argparse.ArgumentParser(description='')
parser.add_argument('--lib', type=str, default='filelist', help='path to data file')
parser.add_argument('--normdata', type=str, default='', help='path to normdata file if necessary')
parser.add_argument('--output', type=str, default='.', help='name of output directory')
parser.add_argument('--features_name', type=str, default='features.csv', help='name of features csv file')
parser.add_argument('--predictions_name', type=str, default='predictions.csv', help='name of predictions csv file')
parser.add_argument('--model', type=str, default='', help='path to pretrained model')
parser.add_argument('--arch', default='alexnet', choices=model_names, help='model architecture: '+' | '.join(model_names)+' (default: alexnet)')
parser.add_argument('--batch_size', type=int, default=100, help='how many images to sample per slide (default: 100)')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--patchsize', default=224, type=int, help='patch size (default: 224)')
parser.add_argument('--mpp', default=0.5, type=float, help='resolution for patch extraction in mpp (default: 0.5)')
parser.add_argument('--overlap', default=1, type=int, help='overlap int (default: 1)')
parser.add_argument('--erode', action='store_true', help='erode when doing background detection')
parser.add_argument('--prune', action='store_true', help='prune when doing background detection')
parser.add_argument('--min_cc_size', default=10, type=int, help='min cc size (default: 10)')
parser.add_argument('--save_features', action='store_true', help='if you want to save the features')
parser.add_argument('--save_plots', action='store_true', help='if you want to save pdf plots')

def main():
    global args
    args = parser.parse_args()

    #load model
    model = models.resnet34(True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    ch = torch.load(args.model)
    model.load_state_dict(ch['state_dict'])
    model = model.cuda()
    cudnn.benchmark = True

    #normalization
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([transforms.ToTensor(),normalize])

    #load data
    dset = MILdataset(args.lib, trans)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    dset.setmode(1)
    probs = inference(epoch, loader, model)
    maxs = group_max(np.array(dset.slideIDX), probs, len(dset.targets))

    fp = open(os.path.join(args.output, 'predictions.csv'), 'w')
    fp.write('file,target,prediction,probability\n')
    for name, target, prob in zip(dset.slidenames, dset.targets, maxs):
        fp.write('{},{},{},{}\n'.format(name, target, int(prob>=0.5), prob))
    fp.close()

def inference(run, loader, model):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i, input in enumerate(loader):
            print('Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(run+1, args.nepochs, i+1, len(loader)))
            input = input.cuda()
            output = F.softmax(model(input), dim=1)
            probs[i*args.batch_size:i*args.batch_size+input_var.size(0)] = output.detach()[:,1].clone()
    return probs.cpu().numpy()

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
    return list(out)

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
