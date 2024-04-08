import textwrap
import argparse
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from brainsignal.config import *
from brainsignal.util import *
from brainsignal import data
from brainsignal import pipeline


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(textwrap.dedent('''\
        Extract preprocessed+epoched data as described in one or more brainsignal config (YAML) files.'''
    ))
    argparser.add_argument('cfg_paths', nargs='+', help='Path(s) to config (YAML) file(s).')
    argparser.add_argument('-t', '--action_type', default=None, help=textwrap.dedent('''\
        Action type to run (e.g., "preprocess" or "plot"). All missing/stale dependencies will be run as well. If
        None, run the last action defined in the config.'''
    ))
    argparser.add_argument('-i', '--action_ids', nargs='+', default=None, help=textwrap.dedent('''\
        IDs to run for final action_type in the pipeline (e.g. plotting_id). If None, defaults to first ID in list.'''
    ))
    argparser.add_argument('-o', '--overwrite', nargs='?', default=False, help=textwrap.dedent('''\
        Whether to overwrite existing results. If ``False``, will only estimate missing results, leaving old 
        ones in place.
        
        NOTE 1: For safety reasons, brainsignal never deletes files or directories, so ``overwrite``
        will clobber any existing files that need to be rebuilt, but it will leave any other older files in place.
        As a result, the output directory may contain a mix of old and new files. To avoid this, you must delete
        existing directories yourself.
        
        NOTE 2: To ensure correctness of all files, brainsignal contains built-in Make-like dependency checking that
        forces a rebuild for all files downstream from a change. For example, if a preprocessed file is modified, then
        all subsequent actions will be re-run as well (to avoid stale results obtained using missing or modified 
        prerequisites). These dependency-based rebuilds will trigger *even if overwrite is False*, since keeping
        stale results is never correct behavior.\
        '''
    ))
    args = argparser.parse_args()
    action_type_top = args.action_type
    action_ids = args.action_ids
    overwrite = get_overwrite(args.overwrite)

    cfg_paths = args.cfg_paths
    for cfg_path in cfg_paths:
        cfg = get_config(cfg_path)
        output_dir = cfg['output_dir']
        if action_ids is not None:
            action_sequences = [
                get_action_sequence(
                    cfg,
                    action_type=action_type_top,
                    action_id=action_id
                ) for action_id in action_ids]
        else:
            action_sequences = [get_action_sequence(cfg, action_type=action_type_top)]
        for action_sequence in action_sequences:
            deps = [[]]
            for data_info in get_data_info(cfg):
                fif_path = data_info['fif']
                suffix = get_fif_suffix(fif_path)
                subject = basename(fif_path[:len(fif_path) - len(suffix)])

                _deps = [fif_path]
                deps[-1].append(fif_path)

                for action in action_sequence:
                    action_type = action['type']
                    if action_type not in ('preprocess', 'epoch'):
                        continue
                    action_kwargs = action['kwargs']
                    action_id = action['id']
                    output_path = get_path(output_dir, 'output', action_type, action_id, subject=subject)
                    mtime, exists = check_deps(output_path, _deps)
                    stale = mtime == 1
                    if overwrite[action_type] or stale or not exists:
                        do_action = True
                    else:
                        do_action = False

                    if do_action:
                        if action_type == 'preprocess':
                            raw = pipeline.preprocess(
                                cfg['output_dir'],
                                fif_path=data_info['fif'],
                                channel_mask_path=data_info['channel_mask_path'],
                                **action_kwargs
                            )

                        elif action_type == 'epoch':
                            stimulus_table_path = data_info.get(
                                'stimulus_table',
                                infer_stimulus_table_path_from_raw(
                                    raw_path=data_info['fif'],
                                    stimulus_type=action_kwargs.get('stimulus_type', 'event')
                                )
                            )

                            epochs = pipeline.epoch(
                                output_dir,
                                subject,
                                stimulus_table_path=stimulus_table_path,
                                **action_kwargs
                            )
                    else:
                        stderr('%s %s exists for subject %s. Skipping. To force re-run, run with overwrite=True.\n' %
                               (ACTION_VERB_TO_NOUN[action_type], action_id, subject))

                    _deps.append(output_path)
                    deps[-1].append(output_path)

            for action in action_sequence:
                action_type = action['type']
                if action_type not in ('plot',):
                    continue
                action_kwargs = action['kwargs']
                action_id = action['id']
                output_path = get_path(output_dir, 'output', action_type, action_id)
                mtime, exists = check_deps(output_path, deps)
                stale = mtime == 1
                if overwrite[action_type] or stale or not exists:
                    do_action = True
                else:
                    do_action = False

                if do_action:
                    if action_type == 'plot':
                        pipeline.plot(
                            output_dir,
                            **action_kwargs
                        )
                else:
                    stderr('%s %s exists. Skipping. To force re-run, run with overwrite=True.\n' %
                           (ACTION_VERB_TO_NOUN[action_type], action_id))
