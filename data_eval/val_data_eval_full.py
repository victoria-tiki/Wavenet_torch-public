import sys
sys.path.append("/projects/bdoy/vsouzaramos/Wavenet_torch")
from Wavenet_torch.data_generators_torch import *
from Wavenet_torch.train_torch import *
from tqdm import tqdm
import matplotlib.pyplot as plt

# Load the checkpointed state_dict
#checkpoint_dir = "/projects/bdoy/vsouzaramos/WaveNet_training_full_2channel_break/checkpoints/model_lre-3_epoch=69-val_loss=0.02986.ckpt"
#checkpoint_path = os.path.join(checkpoint_dir, 'model_lre-3_epoch=01-val_loss=0.00252.ckpt')
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
#noise_dir = '/scratch/bdao/victoria/WaveNet_data' # For Delta
noise_dir = '/work/hdd/bdao/victoria/WaveNet_data' # For DeltaAI
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
                                 noise_prob=0.7, 
                                 noise_range=None, 
                                 num_workers=num_workers,
                                 )

data_module.setup()
val_dataset = data_module.val_dataset
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

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

np.save("/projects/bdoy/vsouzaramos/new_WaveNet_training/data_eval/predictions.npy", predictions)

def FAR_sim(predictions, ground_truth, threshold):
    sample_rate = 4096  # Hz
    window_size = sample_rate  
    half_window = window_size // 2  # +-0.5s = 2048 

    if np.any(ground_truth):  
        event_indices = np.where(ground_truth == 1)[0]  # Indices where an event exists
        detected_events = 0
        for event_index in event_indices:
            start = max(0, event_index - half_window)  
            end = min(len(predictions), event_index + half_window) 
            max_pred = np.max(predictions[start:end])  # Highest detection score in window
            
            # If any detection exceeds threshold within the event window, count as detected
            if max_pred >= threshold:
                detected_events += 1

        # False alarms: Predictions above threshold that are **not within any event window**
        false_alarms = np.sum((predictions >= threshold) & (ground_truth == 0))

    else:  # No true events, all high predictions are false alarms
        false_alarms = np.sum(predictions >= threshold)

    # Compute FAR over the entire duration
    total_time_seconds = len(predictions) / sample_rate
    far = false_alarms / total_time_seconds if total_time_seconds > 0 else 0

    return far

thresholds = [0.1,0.5,0.8,0.9] #np.linspace(0.1,1,5)
FARs = []

for index in range(len(predictions)):
    prediction_FARs=[]
    for threshold in thresholds:
        prediction_FARs.append(FAR_sim(predictions[index], ground_truth[index], threshold))
    FARs.append(prediction_FARs)

FARs = np.stack(FARs)

tFARs = np.logspace(-1,4,1000) # list of threshold FARs for the plot

sens_vols = []
for threshold_index in range(len(thresholds)):
    detections_for_this_threshold=[]
    for tFAR in tFARs:
        good_detections =  np.sum(FARs[:, threshold_index] < tFAR)
        detections_for_this_threshold.append(good_detections)
    sens_vols.append(detections_for_this_threshold)

#print(sens_vols)
sens_vols = np.stack(sens_vols)

fig,ax=plt.subplots()
ax.set_xscale("log")
ax.set_xlabel("False Alarme Rate (hz)")
ax.set_ylabel("Sensitive Volume (% of total volume)")
for t in range(len(sens_vols)):
    ax.plot(tFARs, sens_vols[t]/len(FARs), label =f"detection threshold: {thresholds[t]}")
ax.legend()
plt.savefig("/projects/bdoy/vsouzaramos/new_WaveNet_training/data_eval/new_plot.pdf", format="pdf")