import datetime
import glob
import os

import mne
import numpy as np
import numpy.linalg as la
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import linalg
from scipy.io import loadmat
from sklearn.preprocessing import minmax_scale


def read_data(subject, path, ext):
    """Find and read EEGLAB EEG epoched data structure from MATLAB.

    Args:
        subject (str): subject id
        path (str): file directory/path 
        ext (str): file extension (usually .mat)

    Raises:
        ValueError: If glob finds less or more than 1 matching files.

    Returns:
        mne.io.eeglab.eeglab.EpochsEEGLAB: mne.Epochs instance
    """
    source = glob.glob(os.path.join(path, f"*{subject}*{ext}"))
    if len(source) == 1:
        print(f"Reading file: {source[0]}")
        print(f"Processing subject {subject}")
        data = mne.io.read_epochs_eeglab(source[0])
        data._data = data._data.astype(np.float64)  # ensure correct type
        return data, source[0]
    else:
        raise ValueError(f"Check source! Found {len(source)} matching files.")


def read_metadata_from_csv(subject, path):
    """Find and read design matrix csv file.

    Args:
        subject (str): subject id
        path (str): file directory/path

        ValueError: If glob finds less or more than 1 matching files.

    Returns:
        pandas.DataFrame: pandas.DataFrame instance
    """
    source = glob.glob(os.path.join(path, f"*{subject}*.csv"))
    if len(source) == 1:
        print(f"Reading csv file: {source[0]}")
        return pd.read_csv(source[0]), source[0]
    else:
        raise ValueError(f"Check csv source! Found {len(source)} matching files.")


def split_odd_even(epochs):
    """Split epochs into two halves based on even/odd indices.

    Args:
        epochs (mne.Epochs): mne.Epochs instance

    Returns:
        mne.Epochs: two mne.Epochs instances.
    """
    return epochs[::2], epochs[1::2]


def cov_singletrial(epochs, win=None, ch_names=None):
    """Compute covariance matrix separately for each epoch, sum the covariance matrices, and return the mean covariance.

    Args:
        epochs (mne.Epochs): mne.Epochs instance
        win (list, optional): Time window to subset or crop. Defaults to None.
        ch_names (list, optional): Channels to subset. Defaults to None.

    Returns:
        An array object: channel by channel covariance  
    """

    if win is None:
        win = [epochs.times[0], epochs.times[-1]]
    if ch_names is None:
        ch_names = epochs.ch_names
    dat = (
        epochs.copy().crop(tmin=win[0], tmax=win[1]).pick_channels(ch_names).get_data()
    )
    datcov = np.zeros((dat.shape[1], dat.shape[1]))
    for e in range(dat.shape[0]):
        datcov += np.cov(dat[e])
    datcov /= dat.shape[0]
    print(f"chan_chan covariance matrix: {datcov.shape}")
    return datcov


def cov_avg(x, win=None, ch_names=None):
    """Compute covariance matrix of the averaged/evoked/ERP data.

    Args:
        x (mne.Epochs or mne.Evoked): mne.Epochs or mne.Evoked instance
        win (list, optional): Time window to subset or crop. Defaults to None.
        ch_names (list, optional): Channels to subset. Defaults to None.

    Returns:
        An array object: channel by channel covariance  
    """
    if win is None:
        win = [x.times[0], x.times[-1]]
    if ch_names is None:
        ch_names = x.ch_names

    if isinstance(x, mne.Evoked):
        dat = x.copy()
    elif len(x) > 1:
        dat = x.copy().average()
    else:
        raise TypeError("Only Epochs or Evoked objects are allowed")

    dat = dat.crop(tmin=win[0], tmax=win[1]).pick_channels(ch_names).data
    datcov = np.cov(dat)
    print(f"chan_chan covariance matrix: {datcov.shape}")
    return datcov


def cov_singletrial_scale(
    epochs, feature, feature_range=(1, 2), win=None, ch_names=None
):
    regressor = minmax_scale(epochs.metadata[feature], feature_range=feature_range)
    assert regressor.shape[0] == len(epochs)

    if win is None:
        win = [epochs.times[0], epochs.times[-1]]
    if ch_names is None:
        ch_names = epochs.ch_names
    dat = (
        epochs.copy().crop(tmin=win[0], tmax=win[1]).pick_channels(ch_names).get_data()
    )
    datcov = np.zeros((dat.shape[1], dat.shape[1]))
    for e in range(dat.shape[0]):
        datcov += np.cov(dat[e]) * regressor[e]
    datcov /= dat.shape[0]
    print(f"chan_chan covariance matrix: {datcov.shape}")
    return datcov


def cov_regularize(array, shrink=0.00):
    """Apply shrinkage regularization to a symmetric covariance matrix

    Args:
        array (np.array): symmetric covariance matrix
        shrink (float, optional): shrinkage factor lambda. Defaults to 0.01.

    Returns:
        array (np.array): Regularized covariance matrix
    """
    assert array.shape[0] == array.shape[1]
    return (1 - shrink) * array + shrink * np.mean(la.eig(array)[0]) * np.eye(
        array.shape[0]
    )


def sort_evals_evecs(evals, evecs, top=None, descend=True):
    if descend:
        idx = evals.argsort()[::-1]
    else:
        idx = evals.argsort()

    evals = evals[idx]
    evecs = evecs[:, idx]

    if top is not None:
        evals = evals[:top]
        evecs = evecs[:, :top]
    return evals, evecs


def load_mxc_data():
    # https://stackoverflow.com/questions/29360777/importing-csv-data-stored-in-a-python-module
    try:
        path = os.path.join(os.path.dirname(__file__), "data", "sampleEEGdata-epo.fif")
        print("Loading sampleEEGdata-epo.fif")
        return mne.read_epochs(path)
    except:
        path = os.path.join(os.path.dirname(__file__), "data", "sampleEEGdata.mat")
        print("Loading sampleEEGdata.mat")
        return mne.io.read_epochs_eeglab(path)


def load_mxc_lf():
    try:
        path = os.path.join(os.path.dirname(__file__), "data", "leadfield.npy")
        lf_dict = np.load(path, allow_pickle=True).item()
        print("Loaded data/leadfield.npy")
    except:
        path = os.path.join(os.path.dirname(__file__), "data", "emptyEEG.mat")
        mxc_data = load_mxc_data()
        dat = loadmat(path)
        lf = dat["lf"]
        lf_dict = {}
        lf_dict["eeg_method"] = lf["EEGMethod"][0, 0][0]
        lf_dict["gain"] = lf["Gain"][0, 0]
        lf_dict["gain_desc"] = "chan_orientation_dipole"
        lf_dict["comment"] = lf["Comment"][0, 0][0]
        lf_dict["head_model_type"] = "surface"
        lf_dict["gridloc"] = lf["GridLoc"][0, 0]
        lf_dict["gridorient"] = lf["GridOrient"][0, 0]
        lf_dict["ch_names"] = mxc_data.ch_names
        lf_dict["info"] = mne.create_info(
            lf_dict["ch_names"], sfreq=1000, ch_types="eeg"
        ).set_montage("standard_1005")
        history = f"converted from Mike X Cohen's leadfield matrix .mat file on {datetime.date.today()} with mne {mne.__version__} and numpy {np.__version__}"
        lf_dict["history"] = history
        # np.save('leadfield.npy', lf_dict)
        # mne.viz.plot_topomap(lf['gain'][:,0,134], lf['info'])
        print("Loaded and converted data/emptyEEG.mat")
    return lf_dict


def topomap(data, info, axes=None, title="", cmap="viridis", contours=False, **kwargs):
    if kwargs.get("vmax") is None:
        clim = np.max(np.abs(data))

    if axes is None:
        fig, axes = plt.subplots()
    axes.set_title(title)
    mne.viz.plot_topomap(
        data,
        info,
        cmap=cmap,
        contours=contours,
        axes=axes,
        vmin=-clim,
        vmax=clim,
        **kwargs,
    )
    return axes


def colorbar(
    data,
    axes,
    position="bottom",
    size="5%",
    pad=0.05,
    multiplier=1e6,
    cmap="viridis",
    label="",
    orientation="horizontal",
    **kwargs,
):
    cblim = np.round(np.max(np.abs(data)) * multiplier, 2)
    divider = make_axes_locatable(axes)
    cbaxes = divider.append_axes(position, size=size, pad=pad)
    return mne.viz.plot_brain_colorbar(
        ax=cbaxes,
        clim={"kind": "value", "lims": [-cblim, 0, cblim]},
        colormap=cmap,
        label=label,
        orientation=orientation,
        **kwargs,
    )


# def read_designmat_EEGstruct(path):
#     dat = loadmat(path)
#     return dat

