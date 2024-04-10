import io
import numpy as np
import pandas as pd
import mne
import scipy.signal

from brainsignal.util import *


GROUPBY_SEP = '||'
GROUP_SEP = '-'


def load_raw(
        fif_path,
        channel_mask_path=None,
        fixation_table_path=None,
):
    raw = get_raw(fif_path)
    suffix = get_fif_suffix(fif_path)
    subject = basename(fif_path[:len(fif_path) - len(suffix)])

    if channel_mask_path is not None:
        if channel_mask_path == 'auto':
            mask_suffix = '_channel_mask.csv'
            mask_map = {}
            mask_dir = join(dirname(__file__), 'resources', 'masks')
            if os.path.exists(mask_dir):
                for mask_file in os.listdir(mask_dir):
                    if mask_file.endswith(mask_suffix):
                        prefix = mask_file[:-len(mask_suffix)]
                        mask_map[prefix] = join(mask_dir, mask_file)
                for prefix in mask_map:
                    if subject.startswith(prefix):
                        channel_mask_path = mask_map[prefix]
            else:
                channel_mask_path = None
        else:
            assert os.path.exists(channel_mask_path), 'Channel mask %s not found.' % channel_mask_path

    if channel_mask_path is not None and os.path.exists(channel_mask_path):
        if channel_mask_path == 'all':
            channels_to_drop = raw.info['ch_names']
        else:
            channel_mask = pd.read_csv(channel_mask_path)
            include = channel_mask.include.astype(bool)
            drop = ~include
            channels_to_drop = channel_mask.channel.values[drop].tolist()
        raw.info['temp'] = dict(channels_to_drop=channels_to_drop)
    else:
        raw.info['temp'] = dict(channels_to_drop=None)

    if fixation_table_path is None:
        fixation_table_path = fif_path[:len(fif_path) - len(suffix)] + '_stim_fixation.csv'
    if os.path.exists(fixation_table_path):
        fixation_table = pd.read_csv(fixation_table_path)
        set_table(raw, fixation_table)

    set_subject(raw, subject)

    return raw


def load_stimulus_table(
    stimulus_table_path,
    stimulus_type=None,
):
    if stimulus_type is None:
        stimulus_type = 'event'
    stimulus_table = pd.read_csv(stimulus_table_path)
    if 'onset' not in stimulus_table:
        assert '%s_onset' % stimulus_type in stimulus_table, ('Stimulus table for type "%s" must contain a column '
            'called either "onset" or "%s_onset"' % (stimulus_type, stimulus_type))
        stimulus_table['onset'] = stimulus_table['%s_onset' % stimulus_type].values
    if 'offset' not in stimulus_table:
        assert '%s_offset' % stimulus_type in stimulus_table, ('Stimulus table for type "%s" must contain a column '
            'called either "offset" or "%s_offset"' % (stimulus_type, stimulus_type))
        stimulus_table['offset'] = stimulus_table['%s_offset' % stimulus_type].values
    stimulus_table['duration'] = stimulus_table.offset - stimulus_table.offset

    return stimulus_table


def get_raw(
        path,
):
    raw = mne.io.Raw(path, preload=True)
    raw = raw.crop(raw.times.min(), raw.times.max())

    return raw


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
        picks=None,
        baseline=None,
        normalize_time=False,
        event_duration=None
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

    if normalize_time:
        assert event_duration is not None, 'normalize_time requires event_duration to be specified'
        sfreq = raw.info['sfreq'] * event_duration
        _info = mne.create_info(raw.info['ch_names'], sfreq, ch_types=raw.info.get_channel_types())
        _info['description'] = raw.info['description']
        _info['bads'] = raw.info['bads']
        _raw = mne.io.RawArray(raw.get_data(picks='all'), _info)
        raw = _raw

        event_times = event_times / event_duration

    event_times = np.round(event_times * raw.info['sfreq']).astype(int)
    events[:, 0] = event_times
    events[:, 2] = stimulus_table.index.values

    # Construct epochs
    epochs = mne.Epochs(
        raw,
        events,
        # event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        picks=picks,
        baseline=baseline,
        preload=True
    )
    set_table(epochs, stimulus_table)

    return epochs


def get_epochs_data_by_indices(
        epochs,
        indices
):
    s = epochs[indices].get_data(copy=False)
    t = s.shape[-1]
    s = s.reshape((-1, t))

    return s


def get_epoch_indices_by_label(
        epochs,
        label_columns=None,
        stimulus_table=None
):
    if label_columns is None:
        label_columns = ['condition']
    if stimulus_table is None:
        stimulus_table = get_table(epochs)
    elif not isinstance(label_columns, list):
        label_columns = [label_columns]
    labels = None
    for label_column in label_columns:
        if labels is None:
            labels = stimulus_table[label_column].astype(str)
        else:
            labels += '_' + stimulus_table[label_column].astype(str)
    assert labels is not None, 'At least one label column must be provided'
    label_set = sorted(labels.unique().tolist())
    indices_by_label = {x: labels[labels == x].index.astype(str).tolist() for x in label_set}

    return indices_by_label


def get_evoked(
        paths,
        label_columns=None,
        groupby_columns=None,
        postprocessing_steps=None
):
    evoked = {}
    times = None
    if not isinstance(groupby_columns, list):
        groupby_columns = [groupby_columns]
    for i, path in enumerate(paths):
        epochs = load_epochs(path)
        if postprocessing_steps:
            epochs = process_signal(epochs, postprocessing_steps)
            epochs = epochs.apply_baseline(baseline=epochs.baseline)
        if times is None:
            times = epochs.times
        else:
            assert np.allclose(times, epochs.times), 'Mismatched epoch times at epoch index %d' % i
        stimulus_table = get_table(epochs)
        if groupby_columns != [None]:
            gb_iter = stimulus_table.groupby(groupby_columns)
        else:
            gb_iter = zip([None], [stimulus_table])
        for key, _stimulus_table in gb_iter:
            if key is not None:
                if len(groupby_columns) == 1:
                    key = [key]
                key = [', '.join([str(_x) for _x in x]) if isinstance(x, tuple) else x for x in key]
                key = '_'.join(['%s-%s' % (x, y) for x, y in zip(groupby_columns, key)])
            indices_by_label = get_epoch_indices_by_label(
                epochs,
                label_columns=label_columns,
                stimulus_table=_stimulus_table
            )
            if key not in evoked:
                evoked[key] = {}
            for label in indices_by_label:
                _evoked = get_epochs_data_by_indices(
                    epochs,
                    indices_by_label[label]
                )
                if label not in evoked[key]:
                    evoked[key][label] = []
                evoked[key][label].append(_evoked)
    for key in evoked:
        for label in evoked[key]:
            evoked[key][label] = np.concatenate(evoked[key][label], axis=0)

    assert times is not None, 'At least one epochs file must be provided'

    return evoked, times



def load_epochs(
        path
):
    return mne.read_epochs(path, preload=True)


def get_subject(inst):
    return inst.info['subject_info']['his_id']


def set_subject(inst, val):
    if inst.info['subject_info'] is None:
        inst.info['subject_info'] = {}
    inst.info['subject_info']['his_id'] = val

    return inst


def get_table(inst):
    return pd.read_json(io.StringIO(inst.info['description']))


def set_table(inst, stimulus_table):
    inst.info['description'] = stimulus_table.to_json()

    return inst


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












######################################
#
#  PREPROCESSING STEPS
#
######################################


def drop_bad_channels(
        raw
):
    return raw.drop_channels(raw.info['bads'], on_missing='ignore')


def drop_masked_channels(
        raw
):
    channels_to_drop = raw.info.get('temp', {}).get('channels_to_drop', None)
    subject = get_subject(raw)
    assert channels_to_drop is not None, ('drop_masked_channels called on subject %s with no masked channels. '
        'Either remove this step from the preprocessing or add a mask for this subject.' % subject)

    raw.drop_channels(channels_to_drop, on_missing='ignore')

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


def _psc_transform(x, baseline_mask):
    m = (x * baseline_mask).sum(axis=-1, keepdims=True)  / baseline_mask.sum(axis=-1, keepdims=True)
    x = (x / m) - 1

    return x


def psc_transform(raw):
    fixation_table = get_table(raw)
    s, e = fixation_table.onset.values, fixation_table.offset.values
    times = raw.times[..., None]
    while len(s.shape) < len(times.shape):
        s = s[None, ...]
    while len(e.shape) < len(times.shape):
        e = e[None, ...]
    baseline_mask = np.logical_and(times >= s, times <= e).sum(axis=-1).astype(bool)
    return raw.apply_function(_psc_transform, baseline_mask=baseline_mask, n_jobs=-1)
