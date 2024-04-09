import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from brainsignal.util import *
from brainsignal import data


def preprocess(
        output_dir,
        fif_path,
        preprocessing_id,
        channel_mask_path=None,
        steps=None,
):
    raw = data.load_raw(
        fif_path,
        channel_mask_path=channel_mask_path,
    )
    raw = data.process_signal(raw, steps)
    subject = data.get_subject(raw)

    save_dir = get_path(output_dir, 'subdir', 'preprocess', preprocessing_id, subject=subject)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = get_path(output_dir, 'output', 'preprocess', preprocessing_id, subject=subject)
    raw.save(save_path, overwrite=True)

    return raw


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
    raw_path = get_path(output_dir, 'output', 'preprocess', preprocessing_id, subject=subject)
    raw = data.get_raw(raw_path)
    stimulus_table = data.load_stimulus_table(
        stimulus_table_path,
        stimulus_type=stimulus_type
    )
    if kwargs is None:
        kwargs = {}

    epochs = data.get_epochs(
        raw,
        stimulus_table,
        **kwargs
    )
    if postprocessing_steps:
        epochs = data.process_signal(epochs, postprocessing_steps)
        epochs = epochs.apply_baseline(baseline=kwargs.get('baseline', (None, 0)))

    save_dir = get_path(output_dir, 'subdir', 'epoch', epoching_id, subject=subject)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = get_path(output_dir, 'output', 'epoch', epoching_id, subject=subject)
    epochs.save(save_path, overwrite=True)

    return epochs


def plot(
        output_dir,
        plotting_id,
        epoching_id,
        postprocessing_steps=None,
        baseline=(None, 0),
        label_columns=None,
        groupby_columns=None,
):
    epoching_dir = get_path(output_dir, 'subdir', 'epoch', epoching_id)
    epochs_paths = []
    for i, epoching_file in enumerate([x for x in os.listdir(epoching_dir) if x.endswith(EPOCHS_SUFFIX)]):
        epochs_paths.append(join(epoching_dir, epoching_file))

    evoked, times = data.get_evoked(
        epochs_paths,
        label_columns=label_columns,
        groupby_columns=groupby_columns,
        postprocessing_steps=postprocessing_steps,
        baseline=baseline
    )

    plotting_dir = get_path(output_dir, 'subdir', 'plot', plotting_id)
    if not os.path.exists(plotting_dir):
        os.makedirs(plotting_dir)
    plotting_path = get_path(output_dir, 'image', 'plot', plotting_id)
    output_path = get_path(output_dir, 'output', 'plot', plotting_id)

    output = []
    for group in evoked:
        filename = 'plot'
        if group is not None:
            filename += '_' + group
        filename = plotting_path % filename

        plt.close('all')
        for label in evoked[group]:
            s = evoked[group][label]
            m = s.mean(axis=0)
            e = data.sem(s, axis=0)
            plt.plot(times, m, label=label)
            plt.fill_between(times, m - e, m + e, alpha=0.1)

            row = dict(
                group=group,
                label=label,
                sample_size=s.shape[0]
            )
            output.append(row)

        plt.legend()
        plt.gcf().set_size_inches(7, 5)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.axhline(0., c='k')
        plt.axvline(0., c='k', ls='dotted')
        plt.savefig(filename, dpi=300)

    output = pd.DataFrame(output)
    output.to_csv(output_path, index=False)






