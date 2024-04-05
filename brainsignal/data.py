import numpy as np
import pandas as pd
import mne
import scipy.signal

from brainsignal.util import *


GROUPBY_SEP = '||'
GROUP_SEP = '-'


def load_subject(
        fif,
        stimulus=None,
        channel_mask=None,
        stimulus_type=None,
        drop_bads=False
):
    raw = get_raw(fif)

    suffix = get_fif_suffix(fif)
    subject = basename(fif[:len(fif) - len(suffix)])

    if stimulus is None:
        if stimulus_type is None:
            stimulus_type = 'event'
        path_base = fif[:len(fif) - len(suffix)]
        stimulus = path_base + f'_stim_{stimulus_type}.csv'
    stimulus_table = get_stimulus_table(stimulus, stimulus_type=stimulus_type)

    if channel_mask is None:
        path_base = fif[:len(fif) - len(suffix)]
        channel_mask = path_base + '_channel_mask.csv'
    else:
        assert os.path.exists(channel_mask), 'Channel mask %s not found.' % channel_mask

    masked_channels = set()
    if os.path.exists(channel_mask):
        channel_mask = pd.read_csv(channel_mask)
        include = channel_mask.include.astype(bool)
        drop = ~include
        masked_channels |= set(channel_mask.channel.values[drop].tolist())

    bads = set(raw.info['bads'])
    masked_channels |= bads
    masked_channels = sorted(list(set(masked_channels)))

    if drop_bads:
        raw = raw.drop_channels(masked_channels, on_missing='ignore')

    raw = raw.rename_channels(lambda x, subject=subject: '%s, %s' % (subject, x))
    good_channels = [x for x in raw.info['ch_names'] if x not in masked_channels]

    print(raw._data.shape)

    return raw, stimulus_table, good_channels


def get_raw(
        path,
):
    raw = mne.io.Raw(path, preload=True)
    raw = raw.crop(raw.times.min(), raw.times.max())

    return raw


def get_stimulus_table(
        stimulus_table_path,
        stimulus_type='event'
):
    stimulus_table = pd.read_csv(stimulus_table_path)
    if 'onset' not in stimulus_table:
        assert '%s_onset' % stimulus_type in stimulus_table, ('Stimulus table for type "%s" must contain a column '
            'called either "onset" or "%s_onset"' % stimulus_type)
        stimulus_table['onset'] = stimulus_table['%s_onset' % stimulus_type].values
    if 'offset' not in stimulus_table:
        assert '%s_offset' % stimulus_type in stimulus_table, ('Stimulus table for type "%s" must contain a column '
            'called either "offset" or "%s_offset"' % stimulus_type)
        stimulus_table['offset'] = stimulus_table['%s_offset' % stimulus_type].values
    stimulus_table['duration'] = stimulus_table.offset - stimulus_table.offset

    return stimulus_table


def get_picks(
        raw,
        good_channels=None,
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
    if good_channels is not None:
        _picks = set(mne.pick_channels(good_channels, []).tolist())
        picks &= _picks
    picks = np.array(sorted(list(picks)))

    return picks


def get_epochs(
        raw,
        stimulus_table,
        pad_left_s=0.1,
        pad_right_s=0.,
        duration=None,
        label_columns=None,
        groupby_columns=None,
        picks=None,
        baseline=None
):
    tmin = -pad_left_s
    if duration is None:
        duration = float(stimulus_table.duration.max())
    tmax = duration + pad_right_s
    n_events = len(stimulus_table)

    raw = raw.crop(raw.times.min(), raw.times.max())

    events = np.zeros((n_events, 3), dtype=int)

    # Get event times
    event_times = stimulus_table['onset'].values
    event_times = np.round(event_times * raw.info['sfreq']).astype(int)
    events[:, 0] = event_times

    # Get event labels
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
    if groupby_columns is not None:
        if isinstance(groupby_columns, str):
            groupby_columns = [groupby_columns]
        assert isinstance(groupby_columns, list), 'groupby_columns must be None, a string, or a list'
        groupby_label = None
        for groupby_column in groupby_columns:
            _groupby_label = ('%s=' % groupby_column) + stimulus_table[groupby_column].astype(str)
            if groupby_label is None:
                groupby_label = _groupby_label
            else:
                groupby_label += GROUP_SEP + _groupby_label
        label = groupby_label + GROUPBY_SEP + label
    ix2label = sorted(label.unique().tolist())
    label_indices = np.arange(1, len(ix2label) + 1)
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


def save_epochs(
        epochs,
        save_dir,
        filename_base,
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    path = join(save_dir, filename_base + '-epo.fif')
    epochs.save(path, overwrite=True)

    return path


def load_epochs(
        path
):
    return mne.read_epochs(path, preload=True)


def process_signal(inst, steps):
    if steps is None:
        steps = []
    assert isinstance(steps, list), 'steps must be a list'

    for step in steps:
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

        inst = f(inst, **kwargs)

    return inst


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


def notch_filter(raw, freqs=(60,120,180,240), method='fir', **kwargs):
    raw = raw.notch_filter(freqs, method=method, n_jobs=-1, **kwargs)

    return raw


def set_reference(
        raw,
        **kwargs
):
    for ch_type in ('ecog', 'seeg', 'eeg'):
        if len(mne.pick_types(raw.info, exclude='bads', **{ch_type:True})):
            raw = raw.set_eeg_reference(ch_type=ch_type, ref_channels="average", **kwargs)

    return raw


def resample(
        raw,
        sfreq,
        window='hamming',
        **kwargs
):
    return raw.resample(sfreq, window=window, n_jobs=-1, **kwargs)


def hilbert(raw, **kwargs):
    return raw.apply_hilbert(envelope=True, **kwargs)


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


def rms(raw, w=0.1, **kwargs):
    w = int(np.round(w * raw.info['sfreq']))
    return raw.apply_function(_rms, w=w, n_jobs=-1, **kwargs)


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


def smooth(raw, w=0.1, **kwargs):
    w = int(np.round(w * raw.info['sfreq']))
    return raw.apply_function(_smooth, w=w, n_jobs=-1, **kwargs)


def savgol_filter(raw, w=0.25, polyorder=5, **kwargs):
    w = int(np.round(w * raw.info['sfreq']))
    return raw.apply_function(
        scipy.signal.savgol_filter,
        window_length=w,
        n_jobs=-1,
        polyorder=polyorder,
        **kwargs
    )


def _stft(x, sfreq, fmin=None, fmax=None):
    stft = scipy.signal.ShortTimeFFT(np.ones(256), 1, sfreq, fft_mode='centered')
    S = stft.stft(x)
    f = stft.f
    if fmin is None:
        fmin = 0
    if fmax is None:
        fmax = sfreq / 2
    mask = np.logical_and(f >= fmin, f <= fmax)
    out = (S * mask[..., None]) / mask.sum()

    return out


def stft(raw, fmin=None, fmax=None):
    sfreq = raw.info['sfreq']
    return raw.apply_function(_stft, sfreq=sfreq, fmin=fmin, fmax=fmax, n_jobs=4)