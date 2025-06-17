from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import sys
sys.path.append("/projects/bdoy/vsouzaramos/Wavenet_torch")
from Wavenet_torch.data_generators_torch import *
from Wavenet_torch.train_torch import *

# Load the checkpointed state_dict
checkpoint_path = "/projects/bdoy/vsouzaramos/new_WaveNet_training/checkpoints/model_lre-3_epoch=05-val_loss=0.03974.ckpt"
state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))['state_dict'] 

# Remove the "model." prefix from keys
print(checkpoint_path)
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith("model."):
        new_key = key[len("model."):]  # Remove the "model." prefix
    else:
        new_key = key
    new_state_dict[new_key] = value

# Load the modified state_dict into the model
model = full_module()
model.load_state_dict(new_state_dict)

# Move the model to GPU if available
#if torch.cuda.is_available():
#    model = model.to("cuda")

data_dir = "/projects/bdoy/vsouzaramos/kiet_data"
# noise_dir = '/scratch/bdao/victoria/WaveNet_data' # for Delta
noise_dir = '/work/hdd/bdao/victoria/WaveNet_data' # for DeltaAI

gaussian=0
train=0

if train:
    wf_file = data_dir + '/train.hdf5'
else:
    wf_file = data_dir + '/test.hdf5'

batch_size=32
num_workers=8
n_channels=2

data_module = WaveformDataModule(noise_dir, 
                                 data_dir, 
                                 batch_size=batch_size,
                                 n_channels=n_channels,
                                 gaussian=0,
                                 noise_prob=1, 
                                 noise_range=None, 
                                 num_workers=num_workers,
                                 )

data_module.setup()
val_dataset = data_module.val_dataset


# Limit to one Hour
val_subset = Subset(val_dataset, range(3600)) # 3600 one second streams
val_loader = DataLoader(val_subset, batch_size=4, shuffle=False)

predictions = []
ground_truth = []
for inputs, targets in tqdm(val_loader, desc="Validating", leave=True):
    with torch.no_grad():
        outputs = model(inputs)
    # Collect predictions and ground-truth labels
    predictions.append(outputs.detach().numpy())
    ground_truth.append(targets.detach().numpy())  # Access ground-truth labels

# Concatenate results into arrays
predictions = np.concatenate(predictions).squeeze(axis=-1)
ground_truth = np.concatenate(ground_truth).squeeze(axis=-1)

np.save("/projects/bdoy/vsouzaramos/new_WaveNet_training/one_hour/predictions.npy", predictions)
np.save("/projects/bdoy/vsouzaramos/new_WaveNet_training/one_hour/ground_truth.npy", ground_truth)