import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils
import numpy as np

# we classify according to both br size (low, medium, large) and rt sensitivity (high, low)
NUM_LABELS = 6  # VIDEO, AUDIO, GAMING, HTTP, VPN, TBD
NUM_SAMPLES = 32 # beware: a too low number (less than 20) will cause an implementation bug to pop up 
T_BIN_SIZE = 300  # time bines - each is 0.1 sec
S_BIN_SIZE = 300 # size bins - each is 1400/300 bytes
cnt = 0


class FlowChunkDataset(Dataset):
    """flow chunks dataset."""

    def __init__(self, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # 6 labels
        # 300 quantized time bins per 30 sec (a bin is 0.1 sec)
        # 300 evenets at each time bins corresponding to receiving a packet with binned size
        self.items = torch.zeros([NUM_SAMPLES, S_BIN_SIZE, T_BIN_SIZE], dtype=torch.int32)
        self.label = torch.zeros([NUM_SAMPLES], dtype=torch.int64)
        for i in range(len(self.items)):
            sample = self.items[i]
            label = torch.randint(0, NUM_LABELS-1, (1,))
            for j in range(T_BIN_SIZE):
                packets_size_received = torch.normal(100*label.item(), 25, size=(1, 200)) # 200 random packets received
                hist = torch.histc (packets_size_received, bins = S_BIN_SIZE, min=0, max = 1400)
                sample[:,j] = hist
            self.label[i] = label
            #print(i, sample.shape)  

        label_weights = 1./np.unique(self.label, return_counts=True)[1]
        label_weights /=  np.sum(label_weights)
        self.weights = torch.DoubleTensor([label_weights[l] for l in self.label])


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #sample = {'label': self.flow_label[idx], 'weights': self.flow_chunks_frame[idx]}
        label = self.label[idx]
        sample = self.items[idx]

        return sample, label


def collate_fn(data):
    
    global cnt
    cnt += 1
    #print(len(data))
    X_batch = torch.tensor(np.array([d[0].numpy() for d in data]))
    y_batch = torch.tensor(np.array([d[1].numpy() for d in data]))
    
    return X_batch, y_batch    

batch_size = 1
train_dataset = FlowChunkDataset()
train_sampler = torch.utils.data.WeightedRandomSampler(train_dataset.items,len(train_dataset.items))
#train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, sampler=train_sampler, collate_fn=collate_fn)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, sampler=None, collate_fn=collate_fn)

test_dataset = FlowChunkDataset()
test_sampler = torch.utils.data.WeightedRandomSampler(train_dataset.items,len(train_dataset.items))
#train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, sampler=train_sampler, collate_fn=collate_fn)
test_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, sampler=None, collate_fn=collate_fn)

print(train_dataset[0])  

import sys
sys.path.append("../s2net")
from utils import plot_spk_rec, plot_mem_rec, generate_random_silence_files
from models import SNN, SpikingConv2DLayer, ReadoutLayer, SurrogateHeaviside, SpikingDenseLayer
from data import SpeechCommandsDataset,Pad, MelSpectrogram, Rescale
from optim import RAdam
dtype = torch.float
device = torch.device("cpu")

spike_fn = SurrogateHeaviside.apply

w_init_std = 0.15
w_init_mean = 0.

layers = []
in_channels = 300
out_channels = 100
input_shape = 300
output_shape = 100 # padding mode is "same"
layers.append(SpikingDenseLayer(input_shape, output_shape,spike_fn,
            w_init_mean=w_init_mean,w_init_std=w_init_std, recurrent=False,lateral_connections=True))


# previous layer output has been flattened
input_shape = 100
output_shape = 6
time_reduction="mean" #mean or max
layers.append(ReadoutLayer(input_shape, output_shape,
                 w_init_mean=w_init_mean, w_init_std=w_init_std, time_reduction=time_reduction))

snn = SNN(layers).to(device, dtype)

#print(iter(train_dataloader))
X_batch, y_batch = next(iter(train_dataloader))
X_batch = X_batch.to(device, dtype)
snn(X_batch)

for i,l in enumerate(snn.layers):
    if isinstance(l, SpikingDenseLayer) or isinstance(l, SpikingConv2DLayer):
        print("Layer {}: average number of spikes={:.4f}".format(i,l.spk_rec_hist.mean()))



lr = 1e-3
weight_decay = 1e-5
reg_loss_coef = 0.1
nb_epochs = 10

params = [{'params':l.w, 'lr':lr, "weight_decay":weight_decay } for i,l in enumerate(snn.layers)]
params += [{'params':l.v, 'lr':lr, "weight_decay":weight_decay} for i,l in enumerate(snn.layers[:-1]) if l.recurrent]
params += [{'params':l.b, 'lr':lr} for i,l in enumerate(snn.layers)]
if snn.layers[-1].time_reduction == "mean":
    params += [{'params':l.beta, 'lr':lr} for i,l in enumerate(snn.layers[:-1])]
elif snn.layers[-1].time_reduction == "max":
    params += [{'params':l.beta, 'lr':lr} for i,l in enumerate(snn.layers)]
else:
    raise ValueError("Readout time recution should be 'max' or 'mean'")
    
optimizer = RAdam(params)
 
gamma = 0.85
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)

hist = train(snn, params, optimizer, train_dataloader, test_dataloader, reg_loss_coef, nb_epochs=nb_epochs,
                  scheduler=scheduler, warmup_epochs=1)


test_accuracy = compute_classification_accuracy(snn, test_dataloader)
print("Test accuracy=%.3f"%(test_accuracy))