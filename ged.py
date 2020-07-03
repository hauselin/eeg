import mne
import numpy as np
from . import utils
from scipy import linalg
import os
from datetime import datetime
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt


class ged(object):
    """Class ged for fitting models to neural time series data in mne"""
    def __init__(
        self,
        model='singletrial',  # singletrial, average, regressor
        win_s=None,
        win_r=None,
        ch_names=None,
        regularize=0.0,
        status='',
        verbose=False
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

        self.update_params('model', model)
        self.update_params('regularize', regularize)

    def update_params(self, key, value=None):
        if value is None and key in self.params:
            del self.params[key]
        else:
            self.params[key] = value

    def get_params(self, key=None):
        """Return model parameters dictionary or value of a key.

        Args:
            key (str, optional): Dictionary key. If None returns, returns the entire dictionary. Defaults to None.

        Returns:
            dict if key is True, value of key if key is not None
        """
        if key is None:
            return self.params
        else:
            return self.params.get(key)

    def fit(
        self,
        epochs_s,
        epochs_r=None,
        regularize=0.0,
        feature=None,
        feature_range=(1, 2)
    ):
        if epochs_r is None:  # if no epochs_r provided, epochs_r is epochs_s
            epochs_r = epochs_s.copy()

        # update params
        self.update_params("info", epochs_s.info)
        self.update_params("regularize", regularize)
        self.update_params("epoch_times", epochs_s.times)
        self.update_params("epoch_pnts", epochs_s.times.shape[0])
        self.sfreq = epochs_s.info['sfreq']

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
        epochs_s = epochs_s.copy().crop(tmin=self.win_s[0],
                                        tmax=self.win_s[1]).pick_channels(
                                            self.ch_names
                                        )
        self.update_params("win_s_times", epochs_s.times)
        self.update_params("win_s_pnts", epochs_s.times.shape[0])

        if self.win_r is None:
            self.win_r = [epochs_r.times[0], epochs_r.times[-1]]
        self.update_params("win_r", self.win_r)

        # select times and channesl to compute covariance matrix R
        epochs_r = epochs_r.copy().crop(tmin=self.win_r[0],
                                        tmax=self.win_r[1]).pick_channels(
                                            self.ch_names
                                        )
        self.update_params("win_r_times", epochs_r.times)
        self.update_params("win_r_pnts", epochs_r.times.shape[0])

        # compute covariance matrices
        if self.model == 'singletrial':
            covS = utils.cov_singletrial(epochs_s)
            covR = utils.cov_singletrial(epochs_r)
        elif self.model == 'average':
            covS = utils.cov_avg(epochs_s)
            covR = utils.cov_avg(epochs_r)
        elif self.model == 'regressor':
            assert feature is not None and epochs_s.metadata is not None
            self.update_params('feature', feature)
            self.update_params('feature_value', epochs_s.metadata[feature])
            self.update_params('feature_range', feature_range)
            covS = utils.cov_singletrial_scale(epochs_s, feature, feature_range)
            covR = utils.cov_singletrial(epochs_r)

        if self.regularize:
            covR = utils.cov_regularize(covR, shrink=self.regularize)

        self.covS_ = covS
        self.covR_ = covR

        # perform generalized eigen decomposition
        evals, evecs = linalg.eig(covS, covR)
        if np.any(np.iscomplex(evals)):
            print(
                "GED returned imaginary eigenvalues. Applying 0.01 regularization."
            )
            self.regularize = 0.01
            self.update_params('regularize', self.regularize)
            covR = utils.cov_regularize(covR, shrink=self.regularize)
            evals, evecs = linalg.eig(covS, covR)
            if np.any(np.iscomplex(evals)):
                raise TypeError(
                    "GED still returned imaginary eigenvalues after regularization."
                )
        evals = np.real(evals)

        # sort
        evals, evecs = utils.sort_evals_evecs(evals, evecs)
        self.evals_ = evals
        self.evecs_ = evecs  # already normalized (each vector's norm/length is 1)
        self.evalsperc_ = evals / np.sum(evals) * 100  # percent variance

        # save data
        self.info = epochs_s.info

        # compute activation pattern (flip sign if necessary)
        self.comp_pattern()

        # compute component time series
        self.comp_timeseries(epochs_s_original, covR.shape[0])

        # compute S and R win ERP topography values
        self.Swintopo_ = epochs_s.average().data.mean(axis=1)
        self.Rwintopo_ = epochs_r.average().data.mean(axis=1)

        self.status = f"fitted {datetime.now()}"

    def get_top(self, n):
        evals, evecs = utils.sort_evals_evecs(
            self.evalsperc_, self.evecs_, top=n
        )

        pattern = self.comp_pattern_[:, :n]
        return {"evals": evals, "evecs": evecs, "pattern": pattern}

    def get_pattern(self, index=0):
        return self.get_top(index + 1)['pattern'][:, -1].flatten()

    def get_timeseries(self, index=0):
        ts = self.comp_timeseries_[index, :]
        times = self.get_params('epoch_times')
        return {"timeseries": ts, "times": times}

    def get_comp(self, index):
        pattern = self.get_pattern(index)
        ts = self.get_timeseries(index)
        return {
            "pattern": pattern,
            "timeseries": ts['timeseries'],
            "times": ts['times']
        }

    def flipsign_comp_pattern(self, pattern=None, evecs=None):
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
        self.comp_pattern_ = (self.evecs_.T @ self.covS_).T
        if flipsign:
            self.comp_pattern_, self.evecs_, self.signflip_ = self.flipsign_comp_pattern(
            )
        print("Computing component activation pattern using covS_")
        return self.comp_pattern_

    def comp_timeseries(self, data, n=10):
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
        return self.comp_timeseries_

    def transform_pattern(
        self, data_new, win=None, singletrial=True, flipsign=False
    ):
        # copy data (Epochs or Evoked), pick channels first
        # compute covariance matrix in window (singletrial or not)
        # compute activation pattern
        # flip sign if necessary
        # return new pattern (not saved into self)
        pass

    def transform_timeseries(
        self, data_new, win, singletrial=False, flipsign=False
    ):
        pass

    def plot_eigenspectrum(self, axes=None, n=20, cmap='viridis_r', **kwargs):
        if axes is None:
            fig, axes = plt.subplots()
        evals = self.get_top(n).get('evals')
        sns.scatterplot(
            np.arange(evals.shape[0]),
            evals,
            ax=axes,
            palette=cmap,
            hue=evals,
            **kwargs
        )
        axes.set(
            xlabel=f"Eigen index (top {n})", ylabel='Variance explained (%)'
        )
        axes.set_xticks(np.arange(n, step=5))
        axes.set_title(f"Regularize: {self.get_params('regularize')}", size=10)
        axes.legend_.remove()
        sns.despine(fig=plt.gcf(), ax=axes)
        return axes

    def plot_component(
        self, n=0, tmin=None, tmax=None, cmap='viridis', **kwargs
    ):
        fig, axes = plt.subplots(2, 1)
        dat2plot = self.get_pattern(n)
        # plot activation pattern
        utils.topomap(
            dat2plot, self.get_params('info'), cmap=cmap, axes=axes[0]
        )
        utils.colorbar(dat2plot, axes[0], multiplier=1e10, cmap=cmap)
        axes[0].set_title(f"Component {n}\n{self.get_params('win_s')}", size=10)
        # plot component timeseries
        if tmin is None or tmax is None:
            timewins = []
            timewins.extend(self.get_params('win_s'))
            timewins.extend(self.get_params("win_r"))
            tmin = np.min(timewins)
            tmax = np.max(timewins)
        dat2plot = self.get_comp(n)
        idx2plot = np.where(
            (dat2plot.get('times') >= tmin) & (dat2plot.get('times') <= tmax)
        )
        sns.lineplot(
            dat2plot.get('times')[idx2plot],
            dat2plot.get('timeseries')[idx2plot],
            ax=axes[1],
            lw=1,
            color=sns.color_palette(cmap)[0]
        )
        axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        axes[1].set_xticks(np.arange(tmin, tmax + 0.01, step=0.2))
        axes[1].set(xlabel='Time (s)', ylabel='Component amplitude')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig, axes


def save_ged_model(model, path=None, filename=None):
    if filename is None:
        filename = '_unnamed_model.npz'
    if path is None:
        path = '.'
    Path(path).mkdir(parents=True, exist_ok=True)
    outfile = os.path.join(path, filename)
    # dct = {
    #     "covS_": model.covS_,
    #     "covR_": model.covR_,
    #     "evecs_": model.evecs_,
    #     "evals_": model.evals_,
    #     "evalsperc_": model.evalsperc_,
    #     "signflip_": model.signflip_,
    #     "comp_pattern_": model.comp_pattern_,
    #     "comp_timeseries_": model.comp_timeseries_,
    #     "params": model.params
    # }
    dct = {"model": model}
    np.savez(outfile, **dct)


def load_ged_model(path='.', filename='_unnamed_model.npz'):
    infile = os.path.join(path, filename)
    x = np.load(infile, allow_pickle=True)
    model = x['model'].item()
    # model = ged()  # create empty model
    # model.params = x['params'].item()  # update parameters
    # model.covS_ = x['covS_']
    return model