import yaml
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

from brainsignal.util import *
from brainsignal import data


def preprocess(
        output_dir,
        fif_path,
        preprocessing_id,
        channel_mask_path=None,
        fixation_table_path=None,
        steps=None,
):
    raw = data.load_raw(
        fif_path,
        channel_mask_path=channel_mask_path,
        fixation_table_path=fixation_table_path,
    )
    subject = data.get_subject(raw)

    preprocessing_dir = get_path(output_dir, 'subdir', 'preprocess', preprocessing_id, subject=subject)
    if not os.path.exists(preprocessing_dir):
        os.makedirs(preprocessing_dir)
    kwargs = dict(
        output_dir=output_dir,
        fif_path=fif_path,
        preprocessing_id=preprocessing_id,
        channel_mask_path=channel_mask_path,
        fixation_table_path=fixation_table_path,
        steps=steps
    )
    kwargs_path = get_path(output_dir, 'kwargs', 'preprocess', preprocessing_id, subject=subject)
    with open(kwargs_path, 'w') as f:
        yaml.safe_dump(kwargs, f, sort_keys=False)

    raw = data.process_signal(raw, steps)

    preprocessing_path = get_path(output_dir, 'output', 'preprocess', preprocessing_id, subject=subject)
    raw.save(preprocessing_path, overwrite=True)

    del raw


def epoch(
        output_dir,
        subject,
        epoching_id,
        preprocessing_id,
        stimulus_table_path,
        stimulus_type='event',
        postprocessing_steps=None,
        **kwargs
):
    get_epochs_kwargs = kwargs

    epoching_dir = get_path(output_dir, 'subdir', 'epoch', epoching_id, subject=subject)
    if not os.path.exists(epoching_dir):
        os.makedirs(epoching_dir)
    kwargs = dict(
        output_dir=output_dir,
        subject=subject,
        epoching_id=epoching_id,
        preprocessing_id=preprocessing_id,
        stimulus_table_path=stimulus_table_path,
        postprocessing_steps=postprocessing_steps
    )
    kwargs.update(get_epochs_kwargs)
    kwargs_path = get_path(output_dir, 'kwargs', 'epoch', epoching_id, subject=subject)
    with open(kwargs_path, 'w') as f:
        yaml.safe_dump(kwargs, f, sort_keys=False)

    raw_path = get_path(output_dir, 'output', 'preprocess', preprocessing_id, subject=subject)
    raw = data.get_raw(raw_path)
    stimulus_table = data.load_stimulus_table(
        stimulus_table_path,
        stimulus_type=stimulus_type
    )
    if get_epochs_kwargs is None:
        get_epochs_kwargs = {}

    epochs = data.get_epochs(
        raw,
        stimulus_table,
        postprocessing_steps=postprocessing_steps,
        **get_epochs_kwargs
    )

    epoching_path = get_path(output_dir, 'output', 'epoch', epoching_id, subject=subject)
    epochs.save(epoching_path, overwrite=True)

    del epochs
    del raw


def plot(
        output_dir,
        subjects,
        plotting_id,
        epoching_id,
        normalize_time=False,
        tmin_normalized=None,
        tmax_normalized=None,
        postprocessing_steps=None,
        label_columns=None,
        groupby_columns=None,
        by_sensor=False,
        split_times=None,
        as_spectrogram=False,
        window_length=0.5,
        scale_by_band=False,
        vlim=None,
        baseline=None,
        baseline_mode=None
):
    plotting_dir = get_path(output_dir, 'subdir', 'plot', plotting_id)
    if not os.path.exists(plotting_dir):
        os.makedirs(plotting_dir)
    kwargs = dict(
        output_dir=output_dir,
        plotting_id=plotting_id,
        epoching_id=epoching_id,
        normalize_time=normalize_time,
        tmin_normalized=tmin_normalized,
        tmax_normalized=tmax_normalized,
        postprocessing_steps=postprocessing_steps,
        label_columns=label_columns,
        groupby_columns=groupby_columns,
        by_sensor=by_sensor,
        split_times=split_times,
        as_spectrogram=as_spectrogram,
        window_length=window_length,
        scale_by_band=scale_by_band,
        baseline=baseline,
        baseline_mode=baseline_mode,
        vlim=vlim,
    )
    kwargs_path = get_path(output_dir, 'kwargs', 'plot', plotting_id)
    with open(kwargs_path, 'w') as f:
        yaml.safe_dump(kwargs, f, sort_keys=False)

    plotting_dir = get_path(output_dir, 'subdir', 'epoch', epoching_id)
    epochs_paths = []
    for subject in subjects:
        epoching_file = join(plotting_dir, subject + EPOCHS_SUFFIX)
        assert os.path.exists(epoching_file), 'Non-existent epoching file %s. Cannot plot.' % epoching_file
        epochs_paths.append(epoching_file)

    evoked, times, fmin, fmax = data.get_evoked(
        epochs_paths,
        label_columns=label_columns,
        groupby_columns=groupby_columns,
        by_sensor=by_sensor,
        normalize_time=normalize_time,
        tmin_normalized=tmin_normalized,
        tmax_normalized=tmax_normalized,
        postprocessing_steps=postprocessing_steps,
        as_spectrogram=as_spectrogram,
        window_length=window_length,
        scale_by_band=scale_by_band,
        baseline=baseline,
        baseline_mode=baseline_mode,
    )

    plotting_path = get_path(output_dir, 'image', 'plot', plotting_id)
    output_path = get_path(output_dir, 'output', 'plot', plotting_id)

    if split_times is not None:
        split_times = np.array(split_times, dtype=float)

    if vlim is None:
        vlim = (-np.inf, np.inf)

    output = []
    for group in evoked:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        filename = 'plot'
        if group is not None:
            filename += '_' + group
        filename = plotting_path % filename

        plt.close('all')

        for i, label in enumerate(evoked[group]):
            s = evoked[group][label]
            m = np.nanmean(s, axis=0)
            e = data.sem(s, axis=0)
            row = dict(
                group=group,
                label=label,
                sample_size=s.shape[0]
            )
            output.append(row)

            t = s.shape[-1]

            if as_spectrogram:
                extent = (times[0], times[-1], fmin, fmax)
                plt.close('all')

                vcenter = 0
                vmin = m.min() - 1e-8
                vmax = m.max() - 1e-8
                vmin = max(vmin, vlim[0])
                vmax = min(vmax, vlim[1])
                if vmin < 0 and vmax > 0:
                    bound = max(abs(vmin), abs(vmax))
                    vmin = -bound
                    vmax = bound
                elif vmin < 0.:
                    vmax = 1e-8
                else:  # vmax > 0
                    vmin = -1e-8
                norm = matplotlib.colors.TwoSlopeNorm(
                    vmin=vmin,
                    vcenter=vcenter,
                    vmax=vmax
                )
                pos = plt.imshow(
                    m,
                    extent=extent,
                    aspect='auto',
                    interpolation='hanning',
                    norm=norm,
                    cmap='coolwarm',
                )
                plt.title(label)
                plt.gcf().colorbar(pos, ax=plt.gca(), shrink=0.7)
                plt.gcf().set_size_inches(7, 5)
                plt.savefig(filename[:-4] + '_%s.png' % label, dpi=300)
            else:
                if split_times is None:
                    plt.plot(times[:t], m, label=label, color=colors[i])
                    plt.fill_between(times[:t], m - e, m + e, color=colors[i], alpha=0.1)
                else:
                    split_ix = (times[..., None] >= split_times).sum(axis=-1)
                    split_ix[1:len(split_ix)] -= split_ix[:len(split_ix) - 1]
                    split_ix = np.where(split_ix)[0]
                    _split_times = np.pad(split_times, (1, 0), mode='constant', constant_values=-np.inf)

                    m_bin = []
                    e_bin = []
                    t_bin = []

                    for j, (_times, _s, _split_time) in enumerate(
                            zip(
                                np.split(times, split_ix),
                                np.split(s, split_ix, axis=-1),
                                _split_times
                            )
                    ):
                        _m_bin = np.nanmean(_s)
                        m_bin.append(_m_bin)
                        _m_bin = _m_bin * np.ones_like(_times)
                        _e_bin = np.nanmean(data.sem(_s, axis=0))
                        e_bin.append(_e_bin)
                        _e_bin = _e_bin * np.ones_like(_times)
                        _t_bin = _times.mean()
                        t_bin.append(_t_bin)
                        if np.isfinite(_split_time):
                            plt.axvline(_split_time, c='gray', ls='dotted', alpha=0.2)

                    m_bin = np.array(m_bin)
                    e_bin = np.array(e_bin)
                    plt.plot(t_bin, m_bin, label=label, color=colors[i])
                    plt.fill_between(t_bin, m_bin - e_bin, m_bin + e_bin, color=colors[i], alpha=0.1)

        if not as_spectrogram:
            plt.legend()
            plt.gcf().set_size_inches(7, 5)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.axhline(0., c='k')
            plt.axvline(0., c='k')
            plt.savefig(filename, dpi=300)

    output = pd.DataFrame(output)
    output.to_csv(output_path, index=False)






