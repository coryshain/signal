import numpy as np
import pandas as pd
import mne
import scipy.signal


def get_raw(
        path,
):
    raw = mne.io.Raw(path, preload=True)
    raw = raw.crop(raw.times.min(), raw.times.max())

    return raw


def get_picks(
        raw,
        langloc_path=None
):
    pick_types_kwargs = dict(
        meg=True,
        eeg=True,
        ecog=True,
        seeg=True,
        fnirs=True,
        exclude='bads'
    )
    picks = set(mne.pick_types(raw.info, **pick_types_kwargs).tolist())
    if langloc_path:
        langloc = pd.read_csv(langloc_path)
        sel = langloc.s_vs_n_sig > 0
        ch_names = langloc[sel].channel.tolist()
        _picks = set(mne.pick_channels(ch_names, []).tolist())
        picks &= _picks
    picks = np.array(sorted(list(picks)))

    return picks


def get_stimulus_table(
        stimulus_table_path,
        stimulus_type='event'
):
    stimulus_table = pd.read_csv(stimulus_table_path)
    stimulus_table['time'] = stimulus_table['%s_onset' % stimulus_type].values
    stimulus_table['duration'] = stimulus_table['%s_duration' % stimulus_type].values
    stimulus_table['label_index'] = stimulus_table['%s_index' % stimulus_type]

    return stimulus_table


def epochs(
        raw,
        stimulus_table_path,
        stimulus_type='event',
        pad_left_s=0.1,
        pad_right_s=0.,
        duration=None,
        label_columns=None,
        picks=None,
        baseline=None
):
    stimulus_table = get_stimulus_table(stimulus_table_path, stimulus_type=stimulus_type)
    if stimulus_type == 'event':
        stimulus_table = stimulus_table[stimulus_table.key.str.startswith('word')]

    tmin = -pad_left_s
    if duration is None:
        duration = float(stimulus_table.duration.max())
    tmax = duration + pad_right_s
    n_events = len(stimulus_table)

    raw = raw.crop(raw.times.min(), raw.times.max())

    events = np.zeros((n_events, 3), dtype=int)

    # Get event times
    event_times = stimulus_table['time'].values
    event_times = np.round(event_times * raw.info['sfreq']).astype(int)
    events[:, 0] = event_times

    # Get event labels
    label_indices = stimulus_table['label_index'].values
    if label_columns is None:
        label_columns = ['condition']
    elif not isinstance(label_columns, list):
        label_columns = [label_columns]
    label = None
    for label_column in label_columns:
        if label is None:
            label = stimulus_table[label_column].astype(str)
        else:
            label += '_' + stimulus_table[label_column].astype(str)
    assert label is not None, 'At least one label column must be provided'
    ix2label = label.unique()
    event_id = {x: i for i, x in zip(label_indices, ix2label)}
    label = label.map(event_id)
    events[:, 2] = label.values

    # Construct epochs
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        picks=picks,
        baseline=baseline,
        preload=True
    )

    return epochs


def run_preprocessing(raw, cfg, preprocessing_type='raw'):
    cfg = cfg.get('preprocessing', {}).get(preprocessing_type, [])
    assert isinstance(cfg, list), 'cfg must be a list'

    for step in cfg:
        if isinstance(step, str):
            f = globals()[step]
            kwargs = {}
        elif isinstance(step, dict):
            assert len(step) == 1, 'Only one function allowed per step'
            name = list(step.keys())[0]
            f = globals()[name]
            kwargs = step[name]
        else:
            raise ValueError('Unrecognized preprocessing entry: %s' % step)

        raw = f(raw, **kwargs)

    return raw


def filter(
        raw,
        fmin=None,
        fmax=None,
        method='iir',
        **kwargs
):
    return raw.filter(
        l_freq=fmin,
        h_freq=fmax,
        method=method,
        n_jobs=-1,
        **kwargs
    )


def notch_filter(raw, freqs=(60,120,180,240), method='fir'):
    raw = raw.notch_filter(freqs, method=method, n_jobs=-1)

    return raw


def set_reference(
        raw,
):
    for ch_type in ('ecog', 'seeg', 'eeg'):
        if len(mne.pick_types(raw.info, exclude='bads', **{ch_type:True})):
            raw = raw.set_eeg_reference(ch_type=ch_type, ref_channels="average")

    return raw


def resample(
        raw,
        sfreq,
        window='hamming'
):
    return raw.resample(sfreq, window=window, n_jobs=-1)


def hilbert(raw):
    return raw.apply_hilbert(envelope=True)


def zscore(raw):
    return raw.apply_function(lambda x: (x - x.mean()) / x.std(), n_jobs=-1)


def sem(x, axis=None):
    num = np.std(x, axis=axis, ddof=1)
    n = np.size(x) / np.size(num)
    return  num / np.sqrt(n)


def _rms(x, w):
    T = len(x)
    out = np.zeros_like(x)
    n = np.zeros_like(x)
    for _w in range(-w // 2, w // 2):
        s = slice(max(_w, 0), T + _w)
        _x = np.square(x[s])
        out[s] = out[s] + _x
        n[s] += np.ones_like(_x)

    out = out / n
    out = np.sqrt(out)
    return out


def rms(raw, w=0.1):
    w = int(np.round(w * raw.info['sfreq']))
    return raw.apply_function(_rms, w=w, n_jobs=-1)


def _smooth(x, w):
    T = len(x)
    out = np.zeros_like(x)
    n = np.zeros_like(x)
    for _w in range(-w // 2, w // 2):
        s = slice(max(_w, 0), T + _w)
        _x = x[s]
        out[s] = out[s] + _x
        n[s] += np.ones_like(_x)

    out = out / n
    return out


def smooth(raw, w=0.1):
    w = int(np.round(w * raw.info['sfreq']))
    return raw.apply_function(_smooth, w=w, n_jobs=-1)


def savgol(raw, w=0.25, polyorder=3):
    w = int(np.round(w * raw.info['sfreq']))
    return raw.apply_function(scipy.signal.savgol_filter, window_length=w, n_jobs=-1, polyorder=polyorder)


