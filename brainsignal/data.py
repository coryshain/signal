import io
import numpy as np
import pandas as pd
import mne
import scipy.signal

from matplotlib import pyplot as plt

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


def get_baseline_and_mode(
        baseline='auto',
        mode=None,
        tmin=None,
        tmax=None,
        artifact_window=None
):
    if baseline is None:
        return None, None
    if baseline == 'auto':
        baseline = (None, 0)

    baseline = list(baseline)
    if baseline[0] is None and tmin is not None:
        baseline[0] = tmin
    if baseline[1] is None and tmax is not None:
        baseline[1] = tmax

    if baseline[0] is not None and tmin is not None and artifact_window:
        baseline[0] = max(baseline[0], tmin + artifact_window)
    if baseline[1] is not None and tmax is not None and artifact_window:
        baseline[1] = min(baseline[1], tmax - artifact_window)

    baseline = tuple(baseline)

    if mode is None:
        mode = 'mean'

    return baseline, mode


def get_epochs(
        raw,
        stimulus_table,
        pad_left_s=0.1,
        pad_right_s=0.,
        postprocessing_steps=None,
        n_ticks=None,
        duration=None,
        picks=None,
        baseline=None,
        baseline_mode=None,
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
        del raw
        raw = _raw

        event_times = event_times / event_duration

    event_times = np.round(event_times * raw.info['sfreq']).astype(int)
    events[:, 0] = event_times
    events[:, 2] = stimulus_table.index.values

    # Construct epochs
    epochs = mne.Epochs(
        raw,
        events,
        tmin=tmin,
        tmax=tmax,
        picks=picks,
        baseline=None,
        preload=True
    )
    epochs = process_signal(epochs, postprocessing_steps)
    baseline, baseline_mode = get_baseline_and_mode(
        baseline=baseline,
        mode=baseline_mode,
        tmin=epochs.times.min(),
        tmax=epochs.times.max()
    )
    epochs = apply_baseline(epochs, baseline=baseline, mode=baseline_mode, sfreq=epochs.info['sfreq'])

    if normalize_time:
        # Data must be resampled to a fixed number of steps or else epochs may have variable length
        if not n_ticks:
            n_ticks = 1000
        duration = tmax - tmin
        sfreq = epochs.info['sfreq']
        sfreq_new = n_ticks / duration
        if sfreq != sfreq_new:
            epochs = epochs.resample(sfreq_new)
            epochs = epochs.crop(tmin, tmax, include_tmax=False)

    set_table(epochs, stimulus_table)

    return epochs


def get_epochs_data_by_indices(
        epochs,
        indices,
        baseline=None,
        baseline_mode=None,
        ch_names=None,
):
    epochs = epochs[indices]
    baseline, baseline_mode = get_baseline_and_mode(
        baseline=baseline,
        mode=baseline_mode,
        tmin=epochs.times.min(),
        tmax=epochs.times.max()
    )
    epochs = apply_baseline(epochs, baseline=baseline, mode=baseline_mode, sfreq=epochs.info['sfreq'])
    s = epochs.get_data(copy=False, picks=ch_names)
    t = s.shape[-1]
    evoked = s.reshape((-1, t))
    times = epochs.times

    return evoked, times


def get_epochs_spectrogram_by_indices(
        epochs,
        indices,
        fmin=1,
        fmax=None,
        nfreq=100,
        db=False,
        morlet=False,
        window_length=0.5,
        baseline=None,
        baseline_mode=None,
        scale_by_band=False,
        ch_names=None,
        n_jobs=-1
):
    s = epochs[indices]
    sfreq = epochs.info['sfreq']
    if fmax is None:
        fmax = sfreq / 2 - 1
    freqs = np.linspace(fmin, fmax, nfreq)
    if morlet:
        fn = mne.time_frequency.tfr_morlet
    else:
        fn = mne.time_frequency.tfr_multitaper
    n_cycles = freqs * window_length
    artifact_window = window_length / 2
    power = fn(
        s,
        freqs,
        n_cycles,
        return_itc=False,
        decim=10,
        picks=ch_names,
        average=False,
        n_jobs=n_jobs
    )
    if db:
        power.data = 10 * np.log10(power.data)
    if scale_by_band:
        power.data /= power.data.std(axis=-1, keepdims=True)

    baseline, baseline_mode = get_baseline_and_mode(
        baseline=baseline,
        mode=baseline_mode,
        tmin=power.times.min(),
        tmax=power.times.max(),
        artifact_window=artifact_window
    )
    power = apply_baseline(power, baseline=baseline, mode=baseline_mode, sfreq=epochs.info['sfreq'])

    tmin, tmax = power.times[[0, -1]]
    power = power.crop(tmin + artifact_window, tmax - artifact_window)
    times = power.times
    f, t = power.data.shape[-2:]
    evoked = power.data
    evoked = np.flip(evoked.reshape((-1, f, t)), -2)

    return evoked, times


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
        by_sensor=False,
        postprocessing_steps=None,
        as_spectrogram=False,
        window_length=0.5,
        scale_by_band=False,
        baseline=None,
        baseline_mode=None,
):
    evoked = {}
    times = None
    fmin = 1
    fmax = None
    if not isinstance(groupby_columns, list):
        groupby_columns = [groupby_columns]
    for i, path in enumerate(paths):
        epochs = load_epochs(path)
        if as_spectrogram:
            if fmax is None:
                fmax = epochs.info['sfreq'] / 2  # Nyquist frequency
            else:
                assert fmax == epochs.info['sfreq'] / 2, ('All sampling rates must be equal for spectrogram '
                                                             'extraction. Got %sHz and %sHz.' % (
                                                                fmax * 2, epochs.info['sfreq']))
        if postprocessing_steps:
            epochs = process_signal(epochs, postprocessing_steps)
            baseline, baseline_mode = get_baseline_and_mode(
                baseline=baseline,
                mode=baseline_mode,
                tmin=epochs.times.min(),
                tmax=epochs.times.max()
            )
            epochs = apply_baseline(epochs, baseline=baseline, mode=baseline_mode, sfreq=epochs.info['sfreq'])
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
            for label in indices_by_label:
                if by_sensor:
                    ch_names = epochs.info['ch_names']
                    if key is not None:
                        keys = ['%s-%s' % (key, ch_name) for ch_name in ch_names]
                    else:
                        keys = ch_names
                else:
                    ch_names = [None]
                    keys = [key]
                for _key, ch_name in zip(keys, ch_names):
                    if as_spectrogram:
                        _evoked, _times = get_epochs_spectrogram_by_indices(
                            epochs,
                            indices_by_label[label],
                            fmin=fmin,
                            fmax=fmax,
                            baseline=baseline,
                            baseline_mode=baseline_mode,
                            window_length=window_length,
                            scale_by_band=scale_by_band,
                            ch_names=ch_name,
                        )
                    else:
                        _evoked, _times = get_epochs_data_by_indices(
                            epochs,
                            indices_by_label[label],
                            baseline=baseline,
                            baseline_mode=baseline_mode,
                            ch_names=ch_name
                        )
                    if times is None:
                        times = _times
                    if _key not in evoked:
                        evoked[_key] = {}
                    if label not in evoked[_key]:
                        evoked[_key][label] = []
                    evoked[_key][label].append(_evoked)
        del epochs

    for key in evoked:
        for label in evoked[key]:
            _evoked = evoked[key][label]
            n_times = [x.shape[-1] for x in _evoked]
            n_times = min(n_times)
            _evoked = [x[..., :n_times] for x in _evoked]
            evoked[key][label] = np.concatenate(_evoked, axis=0)

    assert times is not None, 'At least one epochs file must be provided'

    return evoked, times, fmin, fmax


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


def _rescale(data, times, baseline, mode="mean"):
    try:
        return mne.baseline.rescale(data, times, baseline, mode=mode, copy=False, picks=None, verbose=None)
    except ValueError:
        # Get baseline indices
        bmin, bmax = baseline
        if bmin is None:
            imin = 0
        else:
            imin = np.where(times >= bmin)[0]
            if len(imin) == 0:
                raise ValueError(
                    f"bmin is too large ({bmin}), it exceeds the largest time value"
                )
            imin = int(imin[0])
        if bmax is None:
            imax = len(times)
        else:
            imax = np.where(times <= bmax)[0]
            if len(imax) == 0:
                raise ValueError(
                    f"bmax is too small ({bmax}), it is smaller than the smallest time "
                    "value"
                )
            imax = int(imax[-1]) + 1
        if imin >= imax:
            raise ValueError(
                f"Bad rescaling slice ({imin}:{imax}) from time values {bmin}, {bmax}"
            )

        if mode == 'scale':
            def fun(d, imin=imin, imax=imax):
                s = np.std(d[..., imin:imax], axis=-1, keepdims=True)
                d /= s

                return d
        else:
            fun = mode

        fun(data)

        return data


def apply_baseline(inst, baseline=None, mode=None, times=None, sfreq=None):
    if baseline is None:
        return inst

    if times is None:
        times = inst.times
    if sfreq is None:
        sfreq = inst.info["sfreq"]

    baseline, mode = get_baseline_and_mode(
        baseline=baseline,
        mode=mode,
        tmin=times.min(),
        tmax=times.max()
    )

    if isinstance(inst, np.ndarray):
        is_np = True
        arr = inst
    else:
        is_np = False
        if hasattr(inst, 'data'):
            attr = 'data'
        else:
            attr = '_data'
        arr = getattr(inst, attr)
    baseline = mne.baseline._check_baseline(
        baseline, times=times, sfreq=sfreq
    )
    _rescale(arr, times, baseline, mode=mode)

    if not is_np:
        inst.baseline = baseline

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
        n_jobs=-1,
        **kwargs
):
    return raw.filter(
        l_freq=fmin,
        h_freq=fmax,
        method=method,
        n_jobs=n_jobs,
        **kwargs
    )


def notch_filter(raw, freqs=(60,120,180,240), method='fir', n_jobs=-1, **kwargs):
    raw = raw.notch_filter(freqs, method=method, n_jobs=n_jobs, **kwargs)

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
        n_jobs=-1,
        **kwargs
):
    return raw.resample(sfreq, window=window, n_jobs=n_jobs, **kwargs)


def hilbert(raw, **kwargs):
    return raw.apply_hilbert(envelope=True, **kwargs)


def zscore(raw, n_jobs=-1):
    return raw.apply_function(lambda x: (x - x.mean()) / x.std(), n_jobs=n_jobs)


def scale(raw, median=False, n_jobs=-1):
    if median:
        def fn(x):
            m = np.median(x)
            s = np.median(np.abs(x - m))
            return x / s
    else:
        def fn(x):
            return x / x.std()

    return raw.apply_function(fn, n_jobs=n_jobs)


def minmax_normalize(raw, n_jobs=-1):
    def fn(x):
        m, M = x.min(), x.max()
        x = (x - m) / (M - m)

        return x

    return raw.apply_function(fn, n_jobs=n_jobs)


def quantile_normalize(raw, lq=0.4, uq=0.6, n_jobs=-1):
    def fn(x, lq=lq, uq=uq):
        s = np.quantile(x, uq) - np.quantile(x, lq)
        m = np.median(x)
        x = (x - m) / s

        return x

    return raw.apply_function(fn, n_jobs=n_jobs)


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


def rms(raw, w=0.1, n_jobs=-1, **kwargs):
    w = int(np.round(w * raw.info['sfreq']))
    return raw.apply_function(_rms, w=w, n_jobs=n_jobs, **kwargs)


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


def smooth(raw, w=0.1, n_jobs=-1, **kwargs):
    w = int(np.round(w * raw.info['sfreq']))
    return raw.apply_function(_smooth, w=w, n_jobs=n_jobs, **kwargs)


def savgol_filter(raw, window_length=0.5, polyorder=5, n_jobs=-1, **kwargs):
    window_length = int(np.round(window_length * raw.info['sfreq']))
    return raw.apply_function(
        scipy.signal.savgol_filter,
        window_length=window_length,
        n_jobs=n_jobs,
        polyorder=polyorder,
        **kwargs
    )


def _psc_transform(x, baseline_mask):
    m = (x * baseline_mask).sum(axis=-1, keepdims=True)  / baseline_mask.sum(axis=-1, keepdims=True)
    x = (x / m) - 1

    return x


def _detrend(y, polyorder=6):
    x = np.arange(len(y))
    polyfit = np.polyfit(x, y, polyorder)
    polyfit[-1] = 0  # Remove intercept, i.e. last coef
    predicted = np.polyval(polyfit, x)
    detrended = y - predicted

    return detrended


def detrend(raw, n_jobs=-1, **kwargs):
    return raw.apply_function(_detrend, n_jobs=n_jobs, **kwargs)


def psc_transform(raw, n_jobs=-1):
    fixation_table = get_table(raw)
    s, e = fixation_table.onset.values, fixation_table.offset.values
    times = raw.times[..., None]
    while len(s.shape) < len(times.shape):
        s = s[None, ...]
    while len(e.shape) < len(times.shape):
        e = e[None, ...]
    baseline_mask = np.logical_and(times >= s, times <= e).sum(axis=-1).astype(bool)
    return raw.apply_function(_psc_transform, baseline_mask=baseline_mask, n_jobs=n_jobs)


def tfr_average(
        inst,
        fmin=1,
        fmax=None,
        n_freqs=25,
        window_length=0.5,
        baseline=None,
        baseline_mode=None,
        scale_by_band=False,
        agg='mean',
        n_jobs=5
):
    if fmax is None:
        fmax = inst.info['sfreq'] / 2 - 1
    else:
        inst = inst.resample((fmax * 2) + 1)
    s = inst._data
    if len(s.shape) == 2:
        expanded = True
        s = s[None, ...]
    else:
        expanded = False
    sfreq = inst.info['sfreq']
    freqs = np.linspace(fmin, fmax, n_freqs)
    n_cycles = freqs * window_length
    n_pad = np.ceil(window_length * sfreq).astype(int)
    s = np.concatenate([
        np.flip(s[..., :n_pad], axis=-1), s, np.flip(s[..., -n_pad:], axis=-1)
    ], axis=-1)  # Reflection pad to mitigate edge effects, which mess up baselining
    power = mne.time_frequency.tfr_array_multitaper(
        s,
        sfreq,
        freqs,
        n_cycles=n_cycles,
        output='power',
        decim=1,
        n_jobs=n_jobs
    )
    power = power[..., n_pad:-n_pad]
    if expanded:
        power = power[0]

    if scale_by_band:
        power /= power.std(axis=-1, keepdims=True)

    baseline, baseline_mode = get_baseline_and_mode(
        baseline=baseline,
        mode=baseline_mode,
        tmin=inst.times.min(),
        tmax=inst.times.max(),
        artifact_window=window_length / 2
    )
    power = apply_baseline(power, baseline=baseline, mode=baseline_mode, times=inst.times, sfreq=inst.info['sfreq'])

    power = getattr(np, agg)(power, axis=-2)  # Aggregate over the frequency bands

    inst._data = power

    return inst
