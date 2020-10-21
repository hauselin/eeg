import datetime
import glob
import os

import mne
import numpy as np
import numpy.linalg as la
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from scipy import linalg
from scipy.io import loadmat
from sklearn.preprocessing import minmax_scale


def read_data(subject, path, ext):
    """Find and read EEGLAB EEG epoched data structure from MATLAB.

    Args:
        subject (str): subject id
        path (str): directory to look for data
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
        path (str): directory to look for data

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
    """[summary]

    Args:
        epochs ([type]): [description]
        feature ([type]): [description]
        feature_range (tuple, optional): [description]. Defaults to (1, 2).
        win ([type], optional): [description]. Defaults to None.
        ch_names ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
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
        shrink (float, optional): shrinkage factor lambda. Defaults to 0.00.

    Returns:
        array (np.array): Regularized covariance matrix
    """
    assert array.shape[0] == array.shape[1]
    return (1 - shrink) * array + shrink * np.mean(la.eig(array)[0]) * np.eye(
        array.shape[0]
    )


def sort_evals_evecs(evals, evecs, top=None, descend=True):
    """[summary]

    Args:
        evals ([type]): [description]
        evecs ([type]): [description]
        top ([type], optional): [description]. Defaults to None.
        descend (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
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
    """[summary]

    Returns:
        [type]: [description]
    """
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
    """[summary]

    Returns:
        [type]: [description]
    """
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
    """[summary]

    Args:
        data ([type]): [description]
        info ([type]): [description]
        axes ([type], optional): [description]. Defaults to None.
        title (str, optional): [description]. Defaults to "".
        cmap (str, optional): [description]. Defaults to "viridis".
        contours (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
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
    """[summary]

    Args:
        data ([type]): [description]
        axes ([type]): [description]
        position (str, optional): [description]. Defaults to "bottom".
        size (str, optional): [description]. Defaults to "5%".
        pad (float, optional): [description]. Defaults to 0.05.
        multiplier ([type], optional): [description]. Defaults to 1e6.
        cmap (str, optional): [description]. Defaults to "viridis".
        label (str, optional): [description]. Defaults to "".
        orientation (str, optional): [description]. Defaults to "horizontal".

    Returns:
        [type]: [description]
    """
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


def eeglab2mne(subject, source_path, source_ext, save_path, overwrite=True):
    """[summary]

    Args:
        subject ([type]): [description]
        source_path ([type]): [description]
        source_ext ([type]): [description]
        save_path ([type]): [description]
        overwrite (bool, optional): [description]. Defaults to True.
    """
    data, source = read_data(subject, source_path, source_ext)
    data.metadata, path = read_metadata_from_csv(subject, source_path)
    data.info["subject_info"] = {"his_id": subject}
    Path(save_path).mkdir(parents=True, exist_ok=True)
    data.save(
        os.path.join(save_path, f"{subject}-epo.fif"), overwrite=overwrite, fmt="double"
    )


def plot_tf(
    data,
    pick=0,
    ax=None,
    title=None,
    cmap="viridis",
    clim=None,  # tuple (0, 1.0)
    clim_scale=0.7,
    colorbar=True,
    ymarks=True,
    xmarks=True,
    aspect="auto",
    log=True,
    n_ticks=10,
    fontsize=16,
    **kwargs,
):
    assert isinstance(data, mne.time_frequency.tfr.AverageTFR)
    dat = data.data[pick]  # select data for channel (np.array)
    extent = [data.times[0], data.times[-1], data.freqs[0], data.freqs[-1]]
    if clim is None:  # set colorbar limits to 0.7 * max value (suggested by Mike)
        maxval = np.max(np.abs(dat))
        clim = (-maxval * clim_scale, maxval * clim_scale)

    fontsize_title, fontsize_ticklabels = fontsize * 1.05, fontsize * 0.8

    if ax is None:
        fig, ax = plt.subplots()

    cm = ax.imshow(
        dat,
        interpolation="gaussian",
        cmap=cmap,
        origin="lower",
        vmin=clim[0],
        vmax=clim[1],
        extent=extent,
        aspect=aspect,
    )

    if log:
        ax.set_yscale("log")
        ax.set_yticks(
            np.round(np.logspace(np.log10(extent[2]), np.log10(extent[3]), n_ticks), 1)
        )
    else:
        ax.set_yticks(np.round(np.linspace(extent[2], extent[3], n_ticks), 1))

    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_tick_params(which="minor", size=0)
    ax.get_yaxis().set_tick_params(which="major", width=0)
    ax.set_adjustable("box")
    ax.tick_params(axis="x", labelsize=fontsize_ticklabels)
    ax.tick_params(axis="y", labelsize=fontsize_ticklabels)

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.075)
        # colorbar_format = "% 2.2f"
        # cb = plt.colorbar(cm, cax=cax, format=colorbar_format)
        cb = plt.colorbar(cm, cax=cax)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=fontsize_ticklabels)
        # for t in cb.ax.get_yticklabels():
        #     print(t)
        #     t.set_x(3.3)

    if xmarks:
        ax.set_xlabel("Time (s)", fontsize=fontsize)
    else:
        ax.set_xticklabels([])
    if ymarks:
        ax.set_ylabel("Frequency (Hz)", fontsize=fontsize)
    else:
        ax.set_yticklabels([])

    if title:
        ax.set_title(title, fontsize=fontsize_title)

    plt.tight_layout()

    return ax


def tf_avg_freq(data, frange, array=False, keepdims=False):
    """Average across multiple frequencies in EpochsTFR or averageTFR. Collapses the frequency dimension to a singleton dimension so the result is like a filtered ERP/EEG time-series.

    Args:
        data (EpochsTFR, averageTFR): mne's EpochsTFR or averageTFR.
        frange (tuple, list): Frequency window min and max values, (fmin, fmax).
        array (bool, optional): Whether to return the raw data as a numpy.array or as mne object. Defaults to False
        keepdims (bool, optional): Whether to keep or remove the singleton dimension after averaging. Defaults to False.

    Raises:
        TypeError: If data isn't an instance of EpochsTFR or averageTFR.

    Returns:
        numpy.array: N-dimensional numpy array (e.g., epochs_channels_times)
    """
    if isinstance(data, mne.time_frequency.tfr.EpochsTFR):
        axis = 2
    elif isinstance(data, mne.time_frequency.tfr.AverageTFR):
        axis = 1
    else:
        raise TypeError("data must be EpochsTFR or AverageTFR")
    freq_mean = np.mean(frange)

    dat = data.copy().crop(fmin=frange[0], fmax=frange[1])
    out = np.mean(dat.data, axis=axis, keepdims=keepdims)
    print(f"Frequencies: {dat.freqs}")
    print(f"Mean frequency: {freq_mean}")
    print(f"Input data dimensions: {dat.data.shape}")
    print(f"Output array dimensions: {out.shape}")

    if not array:
        if isinstance(data, mne.time_frequency.tfr.EpochsTFR):
            # https://mne.tools/stable/auto_examples/io/plot_objects_from_arrays.html
            out = mne.EpochsArray(
                out, info=data.info, metadata=data.metadata, tmin=data.times[0]
            )
            out.metadata["freq"] = freq_mean
        elif isinstance(data, mne.time_frequency.tfr.AverageTFR):
            # https://mne.tools/0.11/generated/mne.EvokedArray.html
            out = mne.EvokedArray(
                out,
                info=data.info,
                tmin=data.times[0],
                comment=f"Mean freq: {freq_mean}",
            )

    return out

