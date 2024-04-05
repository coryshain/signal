import textwrap
import argparse
import numpy as np

from matplotlib import pyplot as plt

from brainsignal.config import *
from brainsignal.util import *
from brainsignal import data


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(textwrap.dedent('''\
        Extract preprocessed+epoched data as described in one or more brainsignal config (YAML) files.'''
    ))
    argparser.add_argument('cfg_paths', nargs='+', help='Path to config (YAML) file.')
    args = argparser.parse_args()

    cfg_paths = args.cfg_paths
    for cfg_path in cfg_paths:
        cfg = get_config(cfg_path)

        output_dir = cfg['output_dir']
        preprocessing = cfg.get('preprocessing', None)
        epoching = cfg['epoching']
        epoch_postprocessing = epoching.pop('postprocessing', None)
        plot_dir = join(output_dir, 'plots')

        event_id = None
        epochs_list = []
        for load_kwargs in cfg['data']:
            load_kwargs['drop_bads'] = cfg.get('drop_bads', False)
            load_kwargs['stimulus_type'] = cfg.get('stimulus_type', None)
            load_kwargs['channel_mask'] = load_kwargs.get('channel_mask', cfg.get('channel_mask', None))

            raw, stimulus_table, good_channels = data.load_subject(**load_kwargs)
            picks = data.get_picks(raw, good_channels=good_channels)

            if not len(picks):
                stderr('No valid sensors found for file %s. Skipping.\n' % load_kwargs['fif'])
                continue

            raw = data.process_signal(raw, preprocessing)

            epochs = data.get_epochs(
                raw,
                stimulus_table,
                picks=picks,
                **epoching
            )

            if event_id is None:
                event_id = epochs.event_id.copy()
            else:
                assert event_id == epochs.event_id, ('Mismatch between event_id dictionaries. Got %s and %s.' %
                                                     (event_id, epochs.event_id))
            epochs = data.process_signal(epochs, epoch_postprocessing)
            epochs = epochs.apply_baseline(baseline=epoching.get('baseline', (None, 0)))
            epochs_list.append(epochs)

            if epoching.get('dump', False):
                fif = load_kwargs['fif']
                suffix = get_fif_suffix(fif)
                subject = basename(fif[:len(fif) - len(suffix)])
                save_dir = join(output_dir, 'epochs')
                epochs_path = data.save_epochs(
                    epochs,
                    save_dir,
                    subject
                )

        times = None
        evoked = {}
        for _event_id in sorted(list(event_id.keys())):
            _event_id_parsed = _event_id.split(data.GROUPBY_SEP)
            if len(_event_id_parsed) == 1:
                group = None
                _event_id_lab = _event_id
            else:
                group, _event_id_lab = _event_id_parsed

            if group not in evoked:
                evoked[group] = {}
            if _event_id_lab not in evoked[group]:
                evoked[group][_event_id_lab] = []

            for epochs in epochs_list:
                if times is None:
                    times = epochs.times
                s = epochs[_event_id].get_data(copy=False)
                t = s.shape[-1]
                s = s.reshape((-1, t))
                evoked[group][_event_id_lab].append(s)

            evoked[group][_event_id_lab] = np.concatenate(evoked[group][_event_id_lab], axis=0)

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        for group in evoked:
            plt.close('all')
            for _event_id in evoked[group]:
                plot_name = 'evoked'
                if group is not None:
                    plot_name += '_' + group
                plot_name += '.png'

                s = evoked[group][_event_id]
                m = s.mean(axis=0)
                e = data.sem(s, axis=0)
                plt.plot(times, m, label=_event_id)
                plt.fill_between(times, m - e, m + e, alpha=0.1)

            plt.legend()
            plt.gcf().set_size_inches(7, 5)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.axhline(0., c='k')
            plt.axvline(0., c='k', ls='dotted')
            plt.savefig(join(plot_dir, plot_name), dpi=300)