import os
import glob
from datetime import datetime
from pathlib import Path
import copy

import mne
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import linalg

from . import utils


class ged(object):
    """Class ged for fitting models to neural time series data in mne.

    Args:
        model (str): GED-type: singletrial, average, regressor. Defaults to 'singletrial'.
        win_s (tuple): Time window (in seconds) for S covariance matrix. e.g., [0.2, 0.8]. Defaults to None (all timepoints).
        win_r (tuple): Time window (in seconds) for R covariance matrix. e.g., [-0.2, 0.0]. Defaults to None (all timepoints).
        ch_names (list): Channels/sensors to use. Defaults to None (all channels).
        regularize (float, optional): Regularization parameter (0: none, 1: full). Defaults to 0.0.
        status (str): Model fit status. Defaults to "".
        verbose (bool): Print logging messages. Defaults to False.
    """

    def __init__(
        self,
        model="singletrial",
        win_s=None,
        win_r=None,
        ch_names=None,
        regularize=0.0,
        status="",
        verbose=False,
    ):
        self.model = model
        self.win_s = win_s
        self.win_r = win_r
        self.ch_names = ch_names
        self.sfreq = None
        self.regularize = regularize
        self.verbose = verbose
        self.status = status
        self.params = {}

        self.update_params("model", model)
        self.update_params("regularize", regularize)

    def update_params(self, key, value=None):
        """Updates dictionary in instance based on key-value pair. If key exists in dictionary and value is None, that key will be removed from the dictionary.

        Args:
            key (str): Dictionary key.
            value (obj, optional): Value for dictionary key. Defaults to None.
        """
        if value is None and key in self.params:
            del self.params[key]
        else:
            self.params[key] = value

    def get_params(self, key=None):
        """Return model parameters dictionary or value of a key.

        Args:
            key (str, optional): Dictionary key. If None returns, returns the entire dictionary. Defaults to None.

        Returns:
            dict if key is None; value of key if key is not None; NoneType if key not found 
        """
        if key is None:
            return self.params
        else:
            return self.params.get(key)

    def fit(
        self, epochs_s, epochs_r=None, feature=None, feature_range=(1, 2),
    ):
        """Fit model to data with generalized eigendecomposition.

        Args:
            epochs_s (mne.Epochs): mne.Epochs instance
            epochs_r (mne.Epochs, optional): mne.Epochs instance. Defaults to None.
            feature (str, optional): Feature/column in epochs_s.metadata to use for single-trial regression GED. Defaults to None.
            feature_range (tuple, optional): Rescale feature range. Defaults to (1, 2).

        Raises:
            TypeError: If GED results contain imaginary values.
        """
        if epochs_r is None:  # if no epochs_r provided, epochs_r is epochs_s
            epochs_r = epochs_s.copy()

        # update params
        self.update_params("info", epochs_s.info)
        self.update_params("regularize", self.regularize)
        self.update_params("epoch_times", epochs_s.times)
        self.update_params("epoch_pnts", epochs_s.times.shape[0])
        self.sfreq = epochs_s.info["sfreq"]

        if self.ch_names is None:
            self.ch_names = epochs_s.ch_names  # if None, use all channels
        self.update_params("ch_names", self.ch_names)
        self.update_params("nbchan", len(self.ch_names))

        if self.win_s is None:
            self.win_s = [epochs_s.times[0], epochs_s.times[-1]]
        self.update_params("win_s", self.win_s)

        # copy epochs for computing time series later on
        epochs_s_original = epochs_s.copy().pick_channels(self.ch_names)
        # select times and channesl to compute covariance matrix S
        epochs_s = (
            epochs_s.copy()
            .crop(tmin=self.win_s[0], tmax=self.win_s[1])
            .pick_channels(self.ch_names)
        )
        self.update_params("win_s_times", epochs_s.times)
        self.update_params("win_s_pnts", epochs_s.times.shape[0])

        if self.win_r is None:
            self.win_r = [epochs_r.times[0], epochs_r.times[-1]]
        self.update_params("win_r", self.win_r)

        # select times and channels to compute covariance matrix R
        epochs_r = (
            epochs_r.copy()
            .crop(tmin=self.win_r[0], tmax=self.win_r[1])
            .pick_channels(self.ch_names)
        )
        self.update_params("win_r_times", epochs_r.times)
        self.update_params("win_r_pnts", epochs_r.times.shape[0])

        # compute covariance matrices
        if self.model == "singletrial":
            covS = utils.cov_singletrial(epochs_s)
            covR = utils.cov_singletrial(epochs_r)
        elif self.model == "average":
            covS = utils.cov_avg(epochs_s)
            covR = utils.cov_avg(epochs_r)
        elif self.model == "regressor":
            assert feature is not None and epochs_s.metadata is not None
            self.update_params("feature", feature)
            self.update_params("feature_value", epochs_s.metadata[feature])
            self.update_params("feature_range", feature_range)
            covS = utils.cov_singletrial_scale(epochs_s, feature, feature_range)
            covR = utils.cov_singletrial(epochs_r)

        if self.regularize:
            covR = utils.cov_regularize(covR, shrink=self.regularize)

        self.covS_ = covS
        self.covR_ = covR

        # perform generalized eigen decomposition
        evals, evecs = linalg.eig(covS, covR)
        # regularize to fix imaginary values
        if np.any(np.iscomplex(evals)) or np.any(np.iscomplex(evecs)):
            print("GED returned imaginary eigenvalues. Applying 0.01 regularization.")
            self.regularize = 0.01
            self.update_params("regularize", self.regularize)
            covR = utils.cov_regularize(covR, shrink=self.regularize)
            evals, evecs = linalg.eig(covS, covR)
            if np.any(np.iscomplex(evals)):
                raise TypeError(
                    "GED still returned imaginary eigenvalues after regularization."
                )
            if np.any(np.iscomplex(evecs)):
                raise TypeError(
                    "GED still returned imaginary eigenvectors after regularization."
                )
        assert (not np.any(np.iscomplex(evals))) and (not np.any(np.iscomplex(evecs)))
        evals = np.real(evals)
        evecs = np.real(evecs)

        # sort eigenvalues and eigenvectors by descending eigenvalues
        evals, evecs = utils.sort_evals_evecs(evals, evecs)
        self.evals_ = evals
        self.evecs_ = evecs  # already normalized (each vector's norm/length is 1)
        self.evalsperc_ = evals / np.sum(evals) * 100  # percent variance

        self.info = epochs_s.info

        # compute activation pattern (flip sign if necessary)
        self.comp_pattern()

        # compute component time series
        self.comp_timeseries(epochs_s_original, covR.shape[0])

        # compute S and R win ERP topography values
        self.Swintopo_ = epochs_s.average().data.mean(axis=1)
        self.Rwintopo_ = epochs_r.average().data.mean(axis=1)

        # time-log when the analysis was completed
        self.status = f"fitted {datetime.now()}"

    def get_top(self, n):
        """Get the top n eigenvalues, eigenvectors, and activation patterns

        Args:
            n (int): Number of components to return.

        Returns:
            dict: Dictionary with evals, evecs, and patterns
        """
        evals, evecs = utils.sort_evals_evecs(self.evalsperc_, self.evecs_, top=n)

        pattern = self.comp_pattern_[:, :n]
        return {"evals": evals, "evecs": evecs, "pattern": pattern}

    def get_pattern(self, index=0):
        """Get activation pattern for a selected component. 

        Args:
            index (int, optional): Index of component/pattern. Defaults to 0.

        Returns:
            numpy.array: 1D numpy.array.

        Notes:
            Uses get_top(self, n) method.
        """
        return self.get_top(index + 1)["pattern"][:, -1].flatten()

    def get_timeseries(self, index=0):
        ts = self.comp_timeseries_[index, :]
        times = self.get_params("epoch_times")
        return {"timeseries": ts, "times": times}

    def get_comp(self, index):
        """[summary]

        Args:
            index ([type]): [description]

        Returns:
            [type]: [description]
        """
        pattern = self.get_pattern(index)
        ts = self.get_timeseries(index)
        return {
            "pattern": pattern,
            "timeseries": ts["timeseries"],
            "times": ts["times"],
        }

    def flipsign_comp_pattern(self, pattern=None, evecs=None):
        """[summary]

        Args:
            pattern ([type], optional): [description]. Defaults to None.
            evecs ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if pattern is None:
            pattern = self.comp_pattern_
        if evecs is None:
            evecs = self.evecs_
        signflip = []
        # flip sign of each component pattern
        for col in range(pattern.shape[1]):
            colval = pattern[:, col]
            idx = np.abs(colval).argmax()
            sign = np.sign(colval[idx])
            signflip.append(sign)
            pattern[:, col] *= sign
            evecs[:, col] *= sign
        return pattern, evecs, np.array(signflip)

    def comp_pattern(self, flipsign=True):
        """[summary]

        Args:
            flipsign (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        self.comp_pattern_ = (self.evecs_.T @ self.covS_).T
        if flipsign:
            (
                self.comp_pattern_,
                self.evecs_,
                self.signflip_,
            ) = self.flipsign_comp_pattern()
        print("Computing component activation pattern using covS_")
        return self.comp_pattern_

    def comp_timeseries(self, data, n=10):
        """Compute averaged component time series for first n components using matrix multiplication (eigenvector @ data).

        Args:
            data (mne.Epochs, mne.Evoked): mne.Epochs or mne.Evoked instance.
            n (int, optional): Number of components' timeseries to compute. Defaults to 10.

        Raises:
            TypeError: If data isn't an instance of mne.Epochs or mne.Evoked.

        Returns:
            dict: Dictionary with timeseries and times.
        """
        if self.covS_.shape[0] < n:
            n = self.covS_.shape[0]
        print(f"Computing component time series for {n} components/dimensions")

        # evoked
        if isinstance(data, mne.Evoked):
            dat = data.copy()
        elif len(data) > 1:
            dat = data.copy().average()
        else:
            raise TypeError("Only Epochs or Evoked objects are allowed")
        assert self.evecs_.shape[0] == dat.data.shape[0]

        self.comp_timeseries_ = self.evecs_[:, :n].T @ dat.data
        return {"timeseries": self.comp_timeseries_, "times": data.times}

    def plot_eigenspectrum(self, axes=None, n=20, cmap="viridis_r", **kwargs):
        """[summary]

        Args:
            axes ([type], optional): [description]. Defaults to None.
            n (int, optional): [description]. Defaults to 20.
            cmap (str, optional): [description]. Defaults to "viridis_r".

        Returns:
            [type]: [description]
        """
        if axes is None:
            fig, axes = plt.subplots()
        evals = self.get_top(n).get("evals")
        sns.scatterplot(
            np.arange(evals.shape[0]), evals, ax=axes, palette=cmap, hue=evals, **kwargs
        )
        axes.set(xlabel=f"Eigen index (top {n})", ylabel="Variance explained (%)")
        axes.set_xticks(np.arange(n, step=5))
        axes.set_title(f"Regularize: {self.get_params('regularize')}", size=10)
        axes.legend_.remove()
        sns.despine(fig=plt.gcf(), ax=axes)
        return axes

    def plot_component(self, n=0, tmin=None, tmax=None, cmap="viridis", **kwargs):
        """[summary]

        Args:
            n (int, optional): [description]. Defaults to 0.
            tmin ([type], optional): [description]. Defaults to None.
            tmax ([type], optional): [description]. Defaults to None.
            cmap (str, optional): [description]. Defaults to "viridis".

        Returns:
            [type]: [description]
        """
        fig, axes = plt.subplots(2, 1)
        dat2plot = self.get_pattern(n)
        # plot activation pattern
        utils.topomap(dat2plot, self.get_params("info"), cmap=cmap, axes=axes[0])
        utils.colorbar(dat2plot, axes[0], multiplier=1e10, cmap=cmap)
        axes[0].set_title(f"Component {n}\n{self.get_params('win_s')}", size=10)
        # plot component timeseries
        if tmin is None or tmax is None:
            timewins = []
            timewins.extend(self.get_params("win_s"))
            timewins.extend(self.get_params("win_r"))
            tmin = np.min(timewins)
            tmax = np.max(timewins)
        dat2plot = self.get_comp(n)
        idx2plot = np.where(
            (dat2plot.get("times") >= tmin) & (dat2plot.get("times") <= tmax)
        )
        sns.lineplot(
            dat2plot.get("times")[idx2plot],
            dat2plot.get("timeseries")[idx2plot],
            ax=axes[1],
            lw=1,
            color=sns.color_palette(cmap)[0],
        )
        axes[1].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        axes[1].set_xticks(np.arange(tmin, tmax + 0.01, step=0.2))
        axes[1].set(xlabel="Time (s)", ylabel="Component amplitude")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig, axes


def save_ged_model(model, path=None, filename=None):
    """Saves GED results in ununcompressed .npz format with numpy.savez.

    Args:
        model (eeg.ged.ged): eeg.ged.ged instance
        path (str, optional): Directory to save to. Defaults to None (current directory).
        filename (str, optional): Filename. Defaults to None ("_unnamed_model.npz").
    """
    if filename is None:
        filename = "_unnamed_model.npz"
    if path is None:
        path = "."
    Path(path).mkdir(parents=True, exist_ok=True)
    outfile = os.path.join(path, filename)
    dct = {"model": model}
    np.savez(outfile, **dct)


def load_ged_model(path=".", filename="_unnamed_model.npz"):
    """Read GED results saved by save_ged_model.

    Args:
        path (str, optional): Directory to load model from. Defaults to current directory.
        filename (str, optional): Filename. Defaults to "_unnamed_model.npz".

    Returns:
        eeg.ged.ged instance: instance of eeg.ged.ged
    """
    infile = os.path.join(path, filename)
    x = np.load(infile, allow_pickle=True)
    model = x["model"].item()
    return model


def load_ged_models(path, subjects):
    """Load all ged models in .npz files in path/directory.

    Args:
        path (str): Location/path of all models (.npz files).

    Returns:
        list: list containing all models
    """
    models = []
    for s in subjects:
        fname = glob.glob(os.path.join(path, f"*{s}*.npz"))[0]
        fname = fname[(len(path) + 1) :]
        print(f"Loading {fname}")
        m = load_ged_model(path, fname)
        models.append(m)
    return models


def plot_ged_results(
    model,
    comps2plot=(0, 1, 2, 3, 4, 5),
    nrows=3,
    cmap="viridis",
    path=None,
    filename=None,
    figsize=(34, 13),
    fontsize=15,
):
    """[summary]

    Args:
        model ([type]): [description]
        comps2plot (tuple, optional): [description]. Defaults to (0, 1, 2, 3, 4, 5).
        nrows (int, optional): [description]. Defaults to 3.
        cmap (str, optional): Colormap. Defaults to "viridis".
        path ([type], optional): [description]. Defaults to None.
        filename ([type], optional): [description]. Defaults to None.
        figsize (tuple, optional): [description]. Defaults to (34, 13).
        fontsize (int, optional): [description]. Defaults to 15.

    Returns:
        Figure, Axes: Instance of matplotlib Figure and Axes
    """

    if isinstance(comps2plot, int) or len(comps2plot) < 3:
        raise ValueError(
            "comps2plot must be tuple with length greater than 3, e.g., (0, 1, 2)"
        )

    fig, ax = plt.subplots(nrows, len(comps2plot), figsize=figsize)
    # plt.get_current_fig_manager().window.showMaximized()  # maximize figure

    # plot eigenvalue spectrum/screeplot (top left)
    cax = ax[0, 0]
    model.plot_eigenspectrum(cax, cmap=cmap + "_r", n=model.evecs_.shape[0])

    # plot component activation patterns (second row)
    ax_array = ax[1, :]
    for idx, cax in enumerate(ax_array):
        dat2plot = model.get_pattern(comps2plot[idx])
        utils.topomap(dat2plot, model.get_params("info"), cmap=cmap, axes=cax)
        utils.colorbar(dat2plot, cax, multiplier=1e10, cmap=cmap)
        cax.set_title(f"Component {comps2plot[idx]}", size=fontsize)
        if idx == 0:
            cax.set(ylabel=f'S win: {model.get_params("win_s")}')

    # plot component time series (third row)
    timewins = []
    timewins.extend(model.get_params("win_s"))
    timewins.extend(model.get_params("win_r"))
    tmin = np.min(timewins)
    tmax = np.max(timewins)
    ax_array = ax[2, :]
    for idx, cax in enumerate(ax_array):
        dat2plot = model.get_comp(comps2plot[idx])
        idx2plot = np.where(
            (dat2plot.get("times") >= tmin) & (dat2plot.get("times") <= tmax)
        )
        sns.lineplot(
            dat2plot.get("times")[idx2plot],
            dat2plot.get("timeseries")[idx2plot],
            ax=cax,
            lw=0.5,
            color=sns.color_palette(cmap)[0],
        )
        cax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        cax.set_xticks(np.arange(tmin, tmax + 0.01, step=0.2))
        cax.set(xlabel="Time (s)", ylabel="Component amplitude")

    # plot S window topography (top right)
    cax = ax[0, -2]
    utils.topomap(model.Swintopo_, model.get_params("info"), cmap=cmap, axes=cax)
    cax.set_title(f"EEG S win: {model.get_params('win_s')}", size=fontsize)
    utils.colorbar(model.Swintopo_, cax, cmap=cmap)

    # plot R window topography
    cax = ax[0, -1]
    utils.topomap(model.Rwintopo_, model.get_params("info"), cmap=cmap, axes=cax)
    utils.colorbar(model.Rwintopo_, cax, cmap=cmap)
    cax.set_title(f"EEG R win: {model.get_params('win_r')}", size=fontsize)

    # delete unused axes in first row
    idx = -3  # start deleting from the -3 axes on the first row
    while ax[0, idx] is not ax[0, 0]:
        fig.delaxes(ax[0, idx])
        idx -= 1

    fig.suptitle(
        f"Subject: {model.get_params('info')['subject_info']['his_id']}",
        fontsize=fontsize,
    )
    fig.set_size_inches(figsize[0], figsize[1])
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if filename:
        if path is None:
            path = "."
        fig.savefig(os.path.join(path, filename))

    return fig, ax


def transform_pattern(model, data, win=None, singletrial=True, flipsign=False):
    """[summary]

    Args:
        model ([type]): [description]
        data ([type]): [description]
        win ([type], optional): [description]. Defaults to None.
        singletrial (bool, optional): [description]. Defaults to True.
        flipsign (bool, optional): [description]. Defaults to False.

    Raises:
        Exception: [description]

    Returns:
        [type]: [description]
    """
    if model.evecs_.shape[0] != len(data.info["ch_names"]):
        raise Exception("model and data must have same channels")
    if singletrial:
        assert len(data.get_data().shape) == 3

    if win is None:
        win = (data.times[0], data.times[-1])
    data = data.copy().crop(*win)

    if singletrial:
        print("Single-trial covariance")
        covS = utils.cov_singletrial(data)
    else:
        print("Average covariance")
        covS = utils.cov_avg(data)

    # transform pattern
    out = copy.deepcopy(model)
    out.covS_ = covS
    out.comp_pattern()
    out.update_params("covS_new_win", win)

    return out


def transform_timeseries(model, data, win=None, singletrial=True, flipsign=False):
    """[summary]

    Args:
        model ([type]): [description]
        data ([type]): [description]
        win ([type], optional): [description]. Defaults to None.
        singletrial (bool, optional): [description]. Defaults to True.
        flipsign (bool, optional): [description]. Defaults to False.

    Raises:
        TypeError: [description]
        Exception: [description]

    Returns:
        [type]: [description]
    """
    assert model.evecs_.shape[0] == len(data.ch_names)
    n_comps = len(data.ch_names)
    result = data.copy()
    if win is not None:
        result.crop(tmin=win[0], tmax=win[1])

    if singletrial and isinstance(result, mne.Evoked):
        raise TypeError("If singletrial=True, provide epochs instead of evoked object.")
    elif not singletrial and isinstance(result, mne.Evoked):
        print(f"Using model method comp_timeseries...")
        compts = model.comp_timeseries(result, n=n_comps)
        assert result._data.shape == compts["timeseries"].shape
        result._data = compts["timeseries"]
    elif singletrial and not isinstance(result, mne.Evoked):
        # ensure we have epochs, which are 3D (epochs_chan_time)
        assert len(result._data.shape) == 3
        print(f"Computing component time-series for {len(result)} epochs.")
        for e in range(len(result)):
            result._data[e] = model.evecs_.T @ result._data[e]
    else:
        raise Exception("Check parameters.")

    return result


def select_comp(models, comps_idx):
    """Select components for different subjects. For each ged instance in models, select the corresponding component in comps_idx. For exampmle, if [ged1, ged2, ged3] and [0, 2, 1], select components 0, 2, 1 from ged1, ged2, and ged respectively. 

    Args:
        models (list): list of ged instances
        comps_idx (list): list of component indices

    Returns:
        : TODO 
    """
    if len(comps_idx) == 1:
        comps_idx *= len(models)
    else:
        assert len(models) == len(comps_idx)
    print(f"Selecting component from each subject: {comps_idx}")
    check = {}

    # get pattern and timeseries for each subject
    evoked_ts = []
    evoked_pattern = []
    inf = models[0].info

    info_ts = mne.create_info(ch_names=["C1"], sfreq=inf["sfreq"], ch_types=["eeg"])
    info_pat = inf
    for i, m in enumerate(models):
        temp_comp = m.get_comp(comps_idx[i])
        temp_times = temp_comp["times"]
        temp_ts = temp_comp["timeseries"].reshape(1, -1)
        # timeseries in evoked array
        e_ts = mne.EvokedArray(
            temp_ts,
            info=info_ts,
            tmin=temp_times[0],
            comment=f"Comp {comps_idx[i]}",
            nave=1,
        )
        e_ts.info["subject_info"] = {}
        e_ts.info["subject_info"]["his_id"] = m.info["subject_info"]["his_id"]
        evoked_ts.append(e_ts)

        # pattern in evoked array
        temp_pat = temp_comp["pattern"].reshape(-1, 1)
        e_pat = mne.EvokedArray(
            temp_pat,
            info=info_pat,
            tmin=np.mean(m.get_params("win_s")),
            comment=f"Comp {comps_idx[i]} Swin: {m.get_params('win_s')}",
            nave=1,
        )
        evoked_pattern.append(e_pat)

        # check
        check[m.info["subject_info"]["his_id"]] = comps_idx[i]

    print(check)
    return evoked_ts, evoked_pattern


def plot_subj_comp(path, subj, comp_idx):
    """Plots the component pattern and time series for a subject. Subject model is loaded from path. Generally used for inspecting components.

    Args:
        path (str): Location/path of all .npz models.
        subj (str): Subject ID or filename.
        comp_idx (int): Component index to plot

    Returns:
        figure, axes, model: matplotlib figure, list of axes, ged instance
    """
    fpath = glob.glob(os.path.join(path, f"*{subj}*.npz"))
    assert len(fpath) == 1
    fname = fpath[0][(len(path) + 1) :]
    print(f"Loading {fname}")
    model = load_ged_model(path, fname)
    f, a = model.plot_component(comp_idx)
    txt = a[0].title.get_text()
    a[0].title.set_text(f"{subj}\n" + txt)
    f.tight_layout()
    return f, a, model
