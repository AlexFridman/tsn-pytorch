# coding: utf-8

# In[1]:


from dataset import TSNDataSet
from models import TSN_LSTM
from opts import parser
from transforms import *

# In[2]:


args_str = """ucf101 RGB     /media/e/vsd/data/ucf101_preprocessed/split_01/file_lists/train_rgb.txt     /media/e/vsd/data/ucf101_preprocessed/split_01/file_lists/test_rgb.txt     --arch BNInception --num_segments 3     --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 80     -b 32 -j 4     --snapshot_pref ucf101_bninception_"""

args = parser.parse_args(args_str.split())

# In[3]:


arch = 'BNInception'
batch_size = 4
clip_gradient = 20.0
consensus_type = 'avg'
dataset = 'ucf101'
dropout = 0.5
epochs = 80
eval_freq = 5
evaluate = False
flow_prefix = ''
gpus = None
k = 3
loss_type = 'nll'
lr = 0.001
lr_steps = [30.0, 60.0]
modality = 'RGB'
momentum = 0.9
no_partialbn = False
num_segments = 3
print_freq = 10
resume = False
snapshot_pref = 'ucf101_bninception_'
start_epoch = 0
train_list = '/media/e/vsd/data/ucf101_preprocessed/split_01/file_lists/train_rgb.txt'
val_list = '/media/e/vsd/data/ucf101_preprocessed/split_01/file_lists/test_rgb.txt'
weight_decay = 0.0005
workers = 4

num_class = 108

checkpoint_path = '/media/d/vsd/tsn-pytorch/ucf101_bninception__rgb_model_best.pth.tar'

# In[4]:


model = TSN_LSTM(num_class, num_segments, modality,
                 base_model=arch,
                 consensus_type=consensus_type, dropout=dropout, partial_bn=not no_partialbn)

crop_size = model.crop_size
scale_size = model.scale_size
input_mean = model.input_mean
input_std = model.input_std
policies = model.get_optim_policies()
train_augmentation = model.get_augmentation()

model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

# In[5]:


checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])

# In[6]:


# if args.modality != 'RGBDiff':
#     normalize = GroupNormalize(input_mean, input_std)
# else:
#     normalize = IdentityTransform()

# if args.modality == 'RGB':
#     data_length = 1
# elif args.modality in ['Flow', 'RGBDiff']:
#     data_length = 5

data_length = 1
data_length = 5

normalize = IdentityTransform()

# In[7]:


val_loader = torch.utils.data.DataLoader(
    TSNDataSet("", val_list, num_segments=num_segments,
               new_length=data_length,
               modality=modality,
               image_tmpl="{:04d}.jpg" if modality in ['RGB', 'RGBDiff'] else "{:04d}.flo",
               random_shift=False,
               transform=torchvision.transforms.Compose([
                   GroupScale(int(scale_size)),
                   GroupCenterCrop(crop_size),
                   Stack(roll=arch == 'BNInception'),
                   ToTorchFormatTensor(div=arch != 'BNInception'),
                   normalize,
               ])),
    batch_size=1, shuffle=False,
    num_workers=workers, pin_memory=True)

# In[8]:


model.eval()

# In[25]:


criterion = torch.nn.CrossEntropyLoss().cuda()

# In[9]:


y_pred = []

for i, (input, target) in enumerate(val_loader):
    target = target.cuda(async=True)
    input_var = torch.autograd.Variable(input, volatile=True)
    target_var = torch.autograd.Variable(target, volatile=True)

    # compute output
    output = model(input_var)
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    y_pred.extend(output)



# In[ ]:
