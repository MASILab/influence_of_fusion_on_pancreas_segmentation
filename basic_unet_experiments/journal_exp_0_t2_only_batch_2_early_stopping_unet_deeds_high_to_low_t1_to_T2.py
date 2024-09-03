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



my_experiment_name = "journal_exp_0_t2_only_fold_{}_batch_2_early_stopping_unet_deeds_high_to_low_t1_to_T2".format(fold)
device_idx = int(which_gpu)

model = monai.networks.nets.BasicUnet(spatial_dims=3, in_channels=1, out_channels=1)






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


        # select relevant df row
        datapoint = self.dataframe.loc[ index ] 


        X = nib.load(datapoint["deeds_high_to_low_t2_reoriented_to_LAS"])
        X = X.get_fdata()


        # clip very top intensities
        percentile_to_clip = 99.999  # Clip the top 1% of intensity values
        clip_value = np.percentile(X, percentile_to_clip)
        X = np.clip(X, 0, clip_value).astype(np.float32)


        # normalize image
        X = (X - X.min()) / (X.max() - X.min())


        X = torch.tensor(X).unsqueeze(0).float() # add the channel dimension

        # GT pancreas segmentation mask 
        mask = nib.load(datapoint["binary_mask"]).get_fdata() 


        mask[mask!=0] = 1

        mask = torch.tensor(mask).unsqueeze(0).float()

        return X, mask









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

for cur_batch, (img_cpu, label_cpu) in tqdm( enumerate(train_gen) ):

    model.train()

    with torch.cuda.amp.autocast():

        img = img_cpu.to(device)
        label = label_cpu.to(device)

        opt.zero_grad()

        pred = model(img)

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
            for cur_val_batch, (img_cpu, label_cpu ) in tqdm( enumerate(val_gen), total=len(val_gen) ):

                with torch.cuda.amp.autocast():

                    img = img_cpu.to(device)
                    label = label_cpu.to(device)

                    pred = model(img)

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
