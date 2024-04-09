import yaml
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

    preprocessing_dir = get_path(output_dir, 'subdir', 'preprocess', preprocessing_id, subject=subject)
    if not os.path.exists(preprocessing_dir):
        os.makedirs(preprocessing_dir)
    kwargs = dict(
        output_dir=output_dir,
        fif_path=fif_path,
        preprocessing_id=preprocessing_id,
        channel_mask_path=channel_mask_path,
        steps=steps
    )
    kwargs_path = get_path(output_dir, 'kwargs', 'preprocess', preprocessing_id, subject=subject)
    with open(kwargs_path, 'w') as f:
        yaml.safe_dump(kwargs, f, sort_keys=False)

    preprocessing_path = get_path(output_dir, 'output', 'preprocess', preprocessing_id, subject=subject)
    raw.save(preprocessing_path, overwrite=True)

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
        **get_epochs_kwargs
    )
    if postprocessing_steps:
        epochs = data.process_signal(epochs, postprocessing_steps)
        epochs = epochs.apply_baseline(baseline=get_epochs_kwargs.get('baseline', (None, 0)))

    epoching_path = get_path(output_dir, 'output', 'epoch', epoching_id, subject=subject)
    epochs.save(epoching_path, overwrite=True)

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
    plotting_dir = get_path(output_dir, 'subdir', 'plot', plotting_id)
    if not os.path.exists(plotting_dir):
        os.makedirs(plotting_dir)
    kwargs = dict(
        output_dir=output_dir,
        plotting_id=plotting_id,
        epoching_id=epoching_id,
        postprocessing_steps=postprocessing_steps,
        baseline=baseline,
        label_columns=label_columns,
        groupby_columns=groupby_columns,
    )
    kwargs_path = get_path(output_dir, 'kwargs', 'plot', plotting_id)
    with open(kwargs_path, 'w') as f:
        yaml.safe_dump(kwargs, f, sort_keys=False)

    plotting_dir = get_path(output_dir, 'subdir', 'epoch', epoching_id)
    epochs_paths = []
    for i, epoching_file in enumerate([x for x in os.listdir(plotting_dir) if x.endswith(EPOCHS_SUFFIX)]):
        epochs_paths.append(join(plotting_dir, epoching_file))

    evoked, times = data.get_evoked(
        epochs_paths,
        label_columns=label_columns,
        groupby_columns=groupby_columns,
        postprocessing_steps=postprocessing_steps,
        baseline=baseline
    )

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






