import torch
from torchsummary import summary
import monai
import pandas as pd
from copy import deepcopy
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import date
import pathlib as pl
import os
import gc
import nibabel as nib
import matplotlib.pyplot as plt

from torch import nn

# import umap

import argparse

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('--fold',
                   help='starting point')

    p.add_argument('--gpu',
                   help='starting point')
    return p


parser = _build_arg_parser()
args = parser.parse_args()

fold = args.fold
which_gpu = args.gpu

if fold == None or which_gpu==None:
    print("Need to provide a fold and gpu")
    exit()

print("Fold:", fold)

print("GPU", which_gpu)



my_experiment_name = "journal_exp_12_upcat_1_fusion_t2_t1_fold_{}_batch_2_early_stopping_unet_deeds_high_to_low_t1_to_T2".format(fold)
device_idx = int(which_gpu)


class Fuse_Output_Of_Upcat_1(nn.Module):

    def __init__(self, t2_model, t1_model):
        super(Fuse_Output_Of_Upcat_1, self).__init__()
        self.t2_model = t2_model
        self.t1_model = t1_model

        # After Fusion, next block needs to accept double the normal input features due to concatenation
        self.t2_model.final_conv = nn.Conv3d(64, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))




    def forward(self, t2_x, t1_x):


        #=========================================
        # T2 encoder
        t2_x0 = self.t2_model.conv_0(t2_x)
        t2_x1 = self.t2_model.down_1(t2_x0)
        t2_x2 = self.t2_model.down_2(t2_x1)
        t2_x3 = self.t2_model.down_3(t2_x2)
        t2_x4 = self.t2_model.down_4(t2_x3)

        t2_u4 = self.t2_model.upcat_4(t2_x4, t2_x3)
        t2_u3 = self.t2_model.upcat_3(t2_u4, t2_x2)
        t2_u2 = self.t2_model.upcat_2(t2_u3, t2_x1)
        t2_u1 = self.t2_model.upcat_1(t2_u2, t2_x0)


        #=========================================
        # T1 encoder
        t1_x0 = self.t1_model.conv_0(t1_x)
        t1_x1 = self.t1_model.down_1(t1_x0)
        t1_x2 = self.t1_model.down_2(t1_x1)
        t1_x3 = self.t1_model.down_3(t1_x2)
        t1_x4 = self.t1_model.down_4(t1_x3)

        t1_u4 = self.t1_model.upcat_4(t1_x4, t1_x3)
        t1_u3 = self.t1_model.upcat_3(t1_u4, t1_x2)
        t1_u2 = self.t1_model.upcat_2(t1_u3, t1_x1)
        t1_u1 = self.t1_model.upcat_1(t1_u2, t1_x0)


        #=========================================
        # Fuse and join skips

        concatenated_u1 = torch.cat((t2_u1, t1_u1), dim=1)


        logits = self.t2_model.final_conv(concatenated_u1)

        return logits



t2_model = monai.networks.nets.BasicUnet(spatial_dims=3, in_channels=1, out_channels=1)
t1_model = monai.networks.nets.BasicUnet(spatial_dims=3, in_channels=1, out_channels=1)

model = Fuse_Output_Of_Upcat_1(t2_model, t1_model)






batch_size = 2
lr = 1e-3
n_steps = 4000 * batch_size # Not sure why, but n_steps gets divied by batch size, so the multiplication is a correction =
n_steps *= 2


val_freq = 20





print(my_experiment_name)


if not pl.Path("/nfs/masi/remedilw/journal_fusion_comparison/runs/").exists():
    pl.Path("/nfs/masi/remedilw/journal_fusion_comparison/runs/").mkdirs()

writer = SummaryWriter(log_dir="/nfs/masi/remedilw/journal_fusion_comparison/runs/")


today = date.today()
day_and_time = today.strftime("%b-%d-%Y")


#==========================
#==========================
#==========================
#==========================
experiment_name = "experiments/{}_{}".format(my_experiment_name, day_and_time)
#==========================
#==========================
#==========================
#==========================


outdir = pl.Path(experiment_name)
if not os.path.exists(outdir):
    os.makedirs(outdir)


tb_outdir = pl.Path("{}".format(experiment_name))
if not os.path.exists(tb_outdir):
    os.makedirs(tb_outdir)





print( "Is cuda available?:", torch.cuda.is_available() )

print( "How many gpus are available?", torch.cuda.device_count() )

print( "Name of my specificed GPU:", torch.cuda.get_device_name(device_idx) )

device = torch.device("cuda:{}".format(device_idx))




model = model.to(device)







train_df = pd.read_csv("/home/remedilw/code/diabetes_pancreas/prepare_mri/journal_pancreas_fusion_data_splits/repeated_scans/f{}_train.csv".format(fold))
val_df = pd.read_csv("/home/remedilw/code/diabetes_pancreas/prepare_mri/journal_pancreas_fusion_data_splits/single_scan/f{}_val.csv".format(fold))








class Dataset(torch.utils.data.Dataset):

    def __init__(self, 
                 dataframe, 
                 n_items, 
                 training):
        
        self.dataframe = deepcopy( dataframe )
        self.n_items = n_items
        self.training = training

    def __len__(self):
        return self.n_items

    def __getitem__(self, _index):


        if self.training:

            # Undersampling - even probability of looking at a patch with any of the labels
            # Uniformly pull a label
            cur_class = np.random.choice( [0, 1] )

            # Randomly grab a row in the dataframe with that label
            index = np.random.choice( self.dataframe[self.dataframe["merged_condition"] == cur_class].index.tolist() )


        else:
            index = _index

        datapoint = self.dataframe.loc[ index ] 



        # Load T1 resampled to T2 resolution using Mrtrix mrgrid regrid
        t1 = nib.load(datapoint["deeds_high_to_low_t1_reoriented_to_LAS_regrid_to_t2_space"])
        t1 = t1.get_fdata()
        percentile_to_clip = 99.999  # Clip the top 1% of intensity values
        clip_value = np.percentile(t1, percentile_to_clip)
        t1 = np.clip(t1, 0, clip_value).astype(np.float32)
        t1 = (t1 - t1.min()) / (t1.max() - t1.min())
        t1 = torch.tensor(t1).unsqueeze(0).float() # add the channel dimension

        # Load T2 
        t2 = nib.load(datapoint["deeds_high_to_low_t2_reoriented_to_LAS"])
        t2 = t2.get_fdata()
        percentile_to_clip = 99.999  # Clip the top 1% of intensity values
        clip_value = np.percentile(t2, percentile_to_clip)
        t2 = np.clip(t2, 0, clip_value).astype(np.float32)
        t2 = (t2 - t2.min()) / (t2.max() - t2.min())
        t2 = torch.tensor(t2).unsqueeze(0).float() # add the channel dimension




        # GT pancreas segmentation mask 
        # mask = np.rot90( nib.load(datapoint["mask"]).get_fdata() ).copy()
        mask = nib.load(datapoint["binary_mask"]).get_fdata() 


        # mask often has a craazy high value rather than 1
        mask[mask!=0] = 1

        mask = torch.tensor(mask).unsqueeze(0).float()



        return t2, t1, mask









train_ds = Dataset(dataframe=train_df, 
                    n_items=n_steps, 
                    training=True)


train_gen = torch.utils.data.DataLoader(
                    train_ds,
                    batch_size=batch_size,
                    shuffle=False, # random sampling already
                    pin_memory=True,
                    num_workers=20,
                    prefetch_factor=3
            )





val_ds = Dataset(dataframe=val_df, 
                    n_items=len(val_df), 
                    training=False)


val_gen = torch.utils.data.DataLoader(
                        val_ds,
                        batch_size=batch_size,
                        shuffle=False, # random sampling already
                        pin_memory=True,
                        num_workers=20,
                        prefetch_factor=3
            )

opt = torch.optim.AdamW(model.parameters(), lr=lr)

loss_obj = monai.losses.DiceLoss(sigmoid=True)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=opt,
        max_lr=lr,
        total_steps=n_steps + 1,
        cycle_momentum=True,
    )
opt.step()

scaler = torch.cuda.amp.GradScaler()







my_current_step = 0

best_val_loss = 99999
patience = 25  # if it doesn't improve by 25 * batch 2 * val_freq 20 = 1000 examples seen
current_patience = 0

for cur_batch, (img0_cpu, img1_cpu, label_cpu) in tqdm( enumerate(train_gen) ):

    model.train()

    with torch.cuda.amp.autocast():

        img0 = img0_cpu.to(device)
        img1 = img1_cpu.to(device)

        label = label_cpu.to(device)

        opt.zero_grad()

        pred = model(img0, img1)

        loss = loss_obj(pred, label)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()


    scheduler.step()



    writer.add_scalars( str(tb_outdir / 'loss'), {
        'train': loss.detach().cpu().numpy().item(),

    }, cur_batch)

    gc.collect()


    if cur_batch % val_freq == 0:
        # torch.save(model.state_dict(), outdir/"epoch_{}_weights.pth".format(cur_batch))

        # run validation
        model.eval()

        # Layer you want to access

        with torch.no_grad():

            avg_val_loss = 0.0
            for cur_val_batch, (img0_cpu, img1_cpu, label_cpu ) in tqdm( enumerate(val_gen), total=len(val_gen) ):

                with torch.cuda.amp.autocast():

                    img0 = img0_cpu.to(device)
                    img1 = img1_cpu.to(device)
                    label = label_cpu.to(device)

                    pred = model(img0, img1)


                    loss = loss_obj(pred, label)

                    val_loss = loss

                    avg_val_loss += val_loss

        # normlaize by the number of batches
        # we are using batch size 1. first batch has cu_val_batch == 0
        # n batches is cur_val_batch + 1 at the end
        avg_val_loss /= cur_val_batch+1

        writer.add_scalars( str(tb_outdir / 'loss'), {
            'val': avg_val_loss,

        }, cur_batch)


        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            current_patience = 0
            # Save the model if you want to
            # torch.save(model.state_dict(), outdir / "best_model.pth")
            torch.save(model.state_dict(), outdir/"best_model.pth")

        else:
            current_patience += 1

        # Check for early stopping
        if current_patience >= patience:
            print(f"Early stopping at batch {cur_batch}. Best validation loss: {best_val_loss}")
            
            writer.flush()

            break



    my_current_step += 1

    if my_current_step >= n_steps - 1:

        writer.flush()

        print("at final step, leaving")
        break

##########################
##########################
##########################



##########################
##########################
##########################
