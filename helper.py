import os, sys
import time
import numpy as np
import warnings
CASCADE_DIR   = "../CascadeTorch"
MODEL_FOLDER  = os.path.join(CASCADE_DIR, "Pretrained_models")
GT_FOLDER     = os.path.join(CASCADE_DIR, "Ground_truth")
DEMO_FOLDER = os.path.join(CASCADE_DIR, "Demo scripts")
if CASCADE_DIR not in sys.path:
    sys.path.insert(0, CASCADE_DIR)

from cascade2p import cascade, utils, checks, config
checks.check_packages()
from cascade2p.utils import plot_dFF_traces, plot_noise_level_distribution, plot_noise_matched_ground_truth

import torch
import torch.nn as nn
import torch.optim as optim
import tempfile


# ---------------------------------------------------------------------------
# Robust config writing.
# CASCADE's config.write_config() truncates the file before writing and cannot
# serialise numpy scalars (e.g. np.int64 from np.minimum), which crashed the
# write mid-way and left an empty/broken config.yaml. This replacement converts
# numpy types to native Python types and writes atomically (temp file + rename),
# so an interrupted or failed write can never corrupt the config.
# ---------------------------------------------------------------------------
def _to_native(o):
    if isinstance(o, np.integer):     return int(o)
    if isinstance(o, np.floating):    return float(o)
    if isinstance(o, np.bool_):       return bool(o)
    if isinstance(o, np.ndarray):     return [_to_native(x) for x in o.tolist()]
    if isinstance(o, (list, tuple)):  return [_to_native(x) for x in o]
    if isinstance(o, dict):           return {k: _to_native(v) for k, v in o.items()}
    return o


def atomic_write_config(config_dict, save_file):
    """Interrupt-safe + numpy-safe drop-in for cascade2p.config.write_config."""
    import ruamel.yaml as yaml
    yml = yaml.YAML(typ='rt')
    tmpl = yml.load(config.config_template)
    for key in config_dict:
        tmpl[key] = _to_native(config_dict[key])
    folder = os.path.dirname(os.path.abspath(save_file))
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, suffix='.yaml.tmp')
    try:
        with os.fdopen(fd, 'w') as file:
            yml.dump(tmpl, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, save_file)          # atomic rename on the same filesystem
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# patch CASCADE's writer so every config write (create_model_folder, train_model,
# and the loop below) goes through the safe version
config.write_config = atomic_write_config


def shallower_model(filter_sizes,filter_numbers,dense_expansion,windowsize,loss_function,optimizer):
  """"
  Defines the model using Torch.
  The model consists of 2 convolutional layers ('conv_filter'), 1 downsampling layers
  ('MaxPooling1D') and 1 dense layer ('Dense').
  To modify the architecture of the network, only the define_model() function needs to be modified.
  Example: model = define_model(filter_sizes,filter_numbers,dense_expansion,windowsize,loss_function,optimizer)
  """
  import torch.nn as nn
  
  class CascadeModel(nn.Module):
      def __init__(self, filter_sizes, filter_numbers, dense_expansion, windowsize):
          super(CascadeModel, self).__init__()
          
          self.conv1 = nn.Conv1d(1, filter_numbers[0], filter_sizes[0], stride=1)
          self.relu1 = nn.ReLU()
          
          self.conv2 = nn.Conv1d(filter_numbers[0], filter_numbers[1], filter_sizes[1])
          self.relu2 = nn.ReLU()
          
          self.pool1 = nn.MaxPool1d(2)
          
          # Dense layer applied per timestep
          self.dense1 = nn.Linear(filter_numbers[1], dense_expansion)
          self.relu3 = nn.ReLU()
          
          # Calculate flattened size for final layer
          flattened_size = self._calculate_flattened_size(windowsize, filter_sizes) * dense_expansion
          
          self.dense2 = nn.Linear(flattened_size, 1)
      
      def _calculate_flattened_size(self, windowsize, filter_sizes):
          size = windowsize
          size = size - (filter_sizes[0] - 1)
          size = size - (filter_sizes[1] - 1)
          size = size // 2
          return size
      
      def forward(self, x):
          x = x.permute(0, 2, 1)
          
          x = self.conv1(x)
          x = self.relu1(x)
          
          x = self.conv2(x)
          x = self.relu2(x)
          
          x = self.pool1(x)
          
          x = x.permute(0, 2, 1)
          
          x = self.dense1(x)
          x = self.relu3(x)
          
          x = x.view(x.size(0), -1)
          
          x = self.dense2(x)
          
          return x
  
  model = CascadeModel(filter_sizes, filter_numbers, dense_expansion, windowsize)
  
  return model



def train_model(
    model_name, model, model_folder="Pretrained_models", ground_truth_folder="Ground_truth", type="CASCADE"
):

    """Train neural network with parameters specified in the config.yaml file in the model folder

    In this function, a model is configured (defined in the input 'model_name': frame rate, noise levels, ground truth datasets, etc.).
    The ground truth is resampled (function 'preprocess_groundtruth_artificial_noise_balanced', defined in "utils.py").
    The network architecture is defined (function 'define_model', defined in "utils.py").
    The thereby defined model is trained with the resampled ground truth data.
    The trained model with its weight and configuration details is saved to disk.

    Parameters
    ----------
    model_name : str
        Name of the model, e.g. 'Universal_30Hz_smoothing100ms'
        This name has to correspond to the folder with the config.yaml file which defines the model parameters

    model_folder: str
        Absolute or relative path, which defines the location of the specified model_name folder
        Default value 'Pretrained_models' assumes a current working directory in the Cascade folder

    ground_truth_folder : str
        Absolute or relative path, which defines the location of the ground truth datasets
        Default value 'Ground_truth'  assumes a current working directory in the Cascade folder

    Returns
    --------
    None
        All results are saved in the folder model_name as .h5 files containing the trained model

    """
    import time

    model_path = os.path.join(model_folder, model_name)
    cfg_file = os.path.join(model_path, "config.yaml")

    print("Manual mode training")

    # check if configuration file can be found
    if not os.path.isfile(cfg_file):
        m = (
            'The configuration file "config.yaml" can not be found at the location "{}".\n'.format(
                os.path.abspath(cfg_file)
            )
            + 'You have provided the model "{}" at the absolute or relative path "{}".\n'.format(
                model_name, model_folder
            )
            + 'Please check if there is a folder for model "{}" at the location "{}".'.format(
                model_name, os.path.abspath(model_folder)
            )
        )
        print(m)
        raise Exception(m)

    # load cfg dictionary from config.yaml file
    cfg = config.read_config(cfg_file)
    verbose = cfg["verbose"]

    if verbose:
        print(
            "Used configuration for model fitting (file {}):\n".format(
                os.path.abspath(cfg_file)
            )
        )
        for key in cfg:
            print("{}:\t{}".format(key, cfg[key]))

        print("\n\nModels will be saved into this folder:", os.path.abspath(model_path))

    # add base folder to selected training datasets
    training_folders = [
        os.path.join(ground_truth_folder, ds) for ds in cfg["training_datasets"]
    ]

    # check if the training datasets can be found
    missing = False
    for folder in training_folders:
        if not os.path.isdir(folder):
            print(
                'The folder "{}" could not be found at the specified location "{}"'.format(
                    folder, os.path.abspath(folder)
                )
            )
            missing = True
    if missing:
        m = (
            'At least one training dataset could not be located.\nThis could mean that the given path "{}" '.format(
                ground_truth_folder
            )
            + "does not specify the correct location or that e.g. a training dataset referenced in the config.yaml file "
            + "contained a typo."
        )
        print(m)
        raise Exception(m)

    start = time.time()
    # Update model fitting status
    cfg["training_finished"] = "Running"
    config.write_config(cfg, os.path.join(model_path, "config.yaml"))

    nr_model_fits = len(cfg["noise_levels"]) * cfg["ensemble_size"]
    print("Fitting a total of {} models:".format(nr_model_fits))
  
    curr_model_nr = 0

    print(training_folders[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for noise_level in cfg["noise_levels"]:
        for ensemble in range(cfg["ensemble_size"]):
            # train 'ensemble_size' (e.g. 5) models for each noise level

            curr_model_nr += 1
            print(
                "\nFitting model {} with noise level {} (total {} out of {}).".format(
                    ensemble + 1, noise_level, curr_model_nr, nr_model_fits
                )
            )

            if cfg["sampling_rate"] > 30:

                windowsize_suggestion = int(np.power(cfg["sampling_rate"] / 30, 0.25) * 64)

                print(
                    "Window size should be enlarged to "
                    + str(windowsize_suggestion)
                    + " time points (if not done already) due to the high calcium imaging sampling rate ("
                    + str(cfg["sampling_rate"])
                    + ")."
                )

            # preprocess dataset to get uniform dataset for training
            X, Y = utils.preprocess_groundtruth_artificial_noise_balanced(
                ground_truth_folders=training_folders,
                before_frac=cfg["before_frac"],
                windowsize=cfg["windowsize"],
                after_frac=1 - cfg["before_frac"],
                noise_level=noise_level,
                sampling_rate=cfg["sampling_rate"],
                smoothing=cfg["smoothing"] * cfg["sampling_rate"],
                omission_list=[],
                permute=1,
                verbose=cfg["verbose"],
                replicas=1,
                causal_kernel=cfg["causal_kernel"],
            )

            if type=="shallow":
                model = shallower_model(
                    filter_sizes=cfg["filter_sizes"],
                    filter_numbers=cfg["filter_numbers"],
                    dense_expansion=cfg["dense_expansion"],
                    windowsize=cfg["windowsize"],
                    loss_function=cfg["loss_function"],
                    optimizer=cfg["optimizer"],
                )

            else:
                model=utils.define_model(
                    filter_sizes=cfg["filter_sizes"],
                    filter_numbers=cfg["filter_numbers"],
                    dense_expansion=cfg["dense_expansion"],
                    windowsize=cfg["windowsize"],
                    loss_function=cfg["loss_function"],
                    optimizer=cfg["optimizer"],
                )


            model = model.to(device)
            print("Device = ", device)


            optimizer = optim.Adagrad(model.parameters(), lr=0.05)
            
            loss_fn = nn.MSELoss()

            cfg["nr_of_epochs"] = int(np.minimum(
                cfg["nr_of_epochs"], int(10 * np.floor(5e6 / len(X)))
            ))

            X_tensor = torch.FloatTensor(X).to(device)
            Y_tensor = torch.FloatTensor(Y).to(device)
            
            dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

            model.train()
            for epoch in range(cfg["nr_of_epochs"]):
                epoch_loss = 0.0
                for batch_X, batch_Y in dataloader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = loss_fn(outputs, batch_Y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                if cfg["verbose"]:
                    print(f"Epoch {epoch+1}/{cfg['nr_of_epochs']}, Loss: {epoch_loss/len(dataloader):.4f}")

            # save model
            file_name = "Model_NoiseLevel_{}_Ensemble_{}.pth".format(
                int(noise_level), ensemble
            )
            torch.save(model.state_dict(), os.path.join(model_path, file_name))
            print("Saved model:", file_name)

    # Update model fitting status
    cfg['training_finished'] = 'Yes'
    config.write_config(cfg, os.path.join( model_path, 'config.yaml' ))

    print("\n\nDone!")
    print("Runtime: {:.0f} min".format((time.time() - start) / 60))
