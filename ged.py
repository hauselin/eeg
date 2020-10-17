import mne
import numpy as np
from . import utils
from scipy import linalg
import os
from datetime import datetime
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib


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
        model="singletrial",  # singletrial, average, regressor
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
        self.status = None
        self.params = {}

        self.update_params("model", model)
        self.update_params("regularize", regularize)

    def update_params(self, key, value=None):
        """[summary]

        Args:
            key ([type]): [description]
            value ([type], optional): [description]. Defaults to None.
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
        if np.any(np.iscomplex(evals)):
            print("GED returned imaginary eigenvalues. Applying 0.01 regularization.")
            self.regularize = 0.01
            self.update_params("regularize", self.regularize)
            covR = utils.cov_regularize(covR, shrink=self.regularize)
            evals, evecs = linalg.eig(covS, covR)
            if np.any(np.iscomplex(evals)):
                raise TypeError(
                    "GED still returned imaginary eigenvalues after regularization."
                )
        evals = np.real(evals)

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
            TypeError: If data isn't an instance of mne.Epochs or mne.Evoked

        Returns:
            dict: Dictionary with timeseries and times.
        """
        if self.covS_.shape[0] < n:
            n = self.covS_.shape[0]
        print(f"Computing component time series for {n} components/dimensions")

        if isinstance(data, mne.Evoked):
            dat = data.copy()
        elif len(data) > 1:
            dat = data.copy().average()
        else:
            raise TypeError("Only Epochs or Evoked objects are allowed")
        assert self.evecs_.shape[0] == dat.data.shape[0]

        self.comp_timeseries_ = self.evecs_[:, :n].T @ dat.data
        return {"timeseries": self.comp_timeseries_, "times": data.times}

    def transform_pattern(self, data, win=None, singletrial=True, flipsign=False):
        # copy data (Epochs or Evoked), pick channels first
        # compute covariance matrix in window (singletrial or not)
        # compute activation pattern
        # flip sign if necessary
        # return new pattern (not saved into self)
        pass

    def transform_timeseries(self, data, win, singletrial=True, flipsign=False):
        pass

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
    """[summary]

    Args:
        model ([type]): [description]
        path ([type], optional): [description]. Defaults to None.
        filename ([type], optional): [description]. Defaults to None.
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
    """[summary]

    Args:
        path (str, optional): [description]. Defaults to ".".
        filename (str, optional): [description]. Defaults to "_unnamed_model.npz".

    Returns:
        [type]: [description]
    """
    infile = os.path.join(path, filename)
    x = np.load(infile, allow_pickle=True)
    model = x["model"].item()
    # model = ged()  # create empty model
    # model.params = x['params'].item()  # update parameters
    # model.covS_ = x['covS_']
    return model


def plot_ged_results(
    model,
    comps2plot=5,
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
        comps2plot (int, optional): [description]. Defaults to 5.
        nrows (int, optional): [description]. Defaults to 3.
        cmap (str, optional): [description]. Defaults to "viridis".
        path ([type], optional): [description]. Defaults to None.
        filename ([type], optional): [description]. Defaults to None.
        figsize (tuple, optional): [description]. Defaults to (34, 13).
        fontsize (int, optional): [description]. Defaults to 15.

    Returns:
        [type]: [description]
    """

    if comps2plot < 3:
        comps2plot = 3

    fig, ax = plt.subplots(nrows, comps2plot, figsize=figsize)
    # plt.get_current_fig_manager().window.showMaximized()  # maximize figure

    # plot eigenvalue spectrum/screeplot (top left)
    cax = ax[0, 0]
    model.plot_eigenspectrum(cax, cmap=cmap + "_r", n=40)

    # plot component activation patterns (second row)
    ax_array = ax[1, :]
    for idx, cax in enumerate(ax_array):
        dat2plot = model.get_pattern(idx)
        utils.topomap(dat2plot, model.get_params("info"), cmap=cmap, axes=cax)
        utils.colorbar(dat2plot, cax, multiplier=1e10, cmap=cmap)
        cax.set_title(f"Component {idx}", size=fontsize)
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
        dat2plot = model.get_comp(idx)
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
        fig.savefig(os.path.join(path, filename))

    return fig, ax

