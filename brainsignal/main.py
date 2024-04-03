import argparse
import numpy as np

import mne
from mne.baseline import rescale
from mne.io import RawArray
from mne.filter import filter_data
from mne.time_frequency import tfr_stockwell, tfr_morlet, tfr_multitaper, morlet
from mne.stats import bootstrap_confidence_interval
from matplotlib import pyplot as plt

from brainsignal.config import *
from brainsignal.util import *
from brainsignal import data

BANDS = dict(
    # delta=(1, 4),
    theta=(4, 8),
    alpha=(8, 12),
    beta=(12, 35),
    gamma=(35, 70),
    # higamma=(70, 149),
)

def stat_fun(x):
    """Return sum of squares."""
    return np.sum(x ** 2, axis=0)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('cfg_path', help='Path to config (YAML) file.')
    args = argparser.parse_args()

    cfg_path = args.cfg_path
    cfg = get_config(cfg_path)

    output_dir = cfg.get('output_dir', 'plots')

    times = None
    evoked = {}
    for path in cfg['data']:
        suffix = get_fif_suffix(path)
        path_base = path[:len(path) - len(suffix)]

        fif_path = path
        raw = data.get_raw(fif_path)

        if cfg.get('lang_only', True):
            langloc_path = path_base + '_langloc.csv'
        else:
            langloc_path = None
        picks = data.get_picks(raw, langloc_path=langloc_path)
        if not len(picks):
            stderr('No valid sensors found for file %s. Skipping.\n' % path)
            continue

        raw = data.run_preprocessing(raw, cfg, preprocessing_type='raw')

        epochs_kwargs = cfg.get('epochs', {})
        stimulus_type = epochs_kwargs.get('stimulus_type', 'event')
        stimulus_table_path = path_base + f'_stim_{stimulus_type}.csv'
        epochs = data.epochs(
            raw,
            stimulus_table_path,
            picks=picks,
            **epochs_kwargs
        )
        epochs = data.run_preprocessing(epochs, cfg, preprocessing_type='epochs')
        epochs = epochs.apply_baseline(baseline=epochs_kwargs.get('baseline', (None, 0)))

        for i, _event_id in enumerate(epochs.event_id):
            if times is None:
                times = epochs.times
            s = epochs[_event_id].get_data(copy=False)
            t = s.shape[-1]
            s = s.reshape((-1, t))
            if _event_id not in evoked:
                evoked[_event_id] = []
            evoked[_event_id].append(s)

    for _event_id in sorted(list(evoked.keys())):
        s = evoked[_event_id]
        s = np.concatenate(s, axis=0)
        m = s.mean(axis=0)
        e = data.sem(s, axis=0)
        plt.plot(times, m, label=_event_id)
        plt.fill_between(times, m-e, m+e, alpha=0.1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.legend()
    plt.gcf().set_size_inches(7, 5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.axhline(0., c='k')
    plt.axvline(0., c='k', ls='dotted')
    plt.savefig(join(output_dir, 'evoked.png'), dpi=300)