import textwrap
import argparse
from pymatreader import read_mat
import numpy as np
import pandas as pd
import mne

from brainsignal.util import *
from brainsignal.evlab.initialize import initialize


SUFFIX = '_crunched.mat'
MATLAB_CMD = textwrap.dedent('''\
    %s -nodisplay -nosplash -nodesktop -wait -r "addpath('%s'); get_ecog('%s', '%s'); exit;"\
''')


def get_matlab_data(path):
    return read_mat(path)['s']


def get_stimulus_data(h5, stimulus_type='event'):
    timing_tables = h5['for_preproc']['trial_timing_raw']
    freq = h5['for_preproc']['sample_freq_raw']

    if stimulus_type in ('trial', 'event'):
        events_table = pd.DataFrame(h5['events_table']).rename(
            dict(
                stimulus_offset='trial_offset'
            ),
            axis=1
        )
        stimulus_data = []
        for i, timing_table in enumerate(timing_tables):
            event_table = events_table.loc[i].copy()

            timing_table = pd.DataFrame(timing_table).rename(
                dict(
                    start='event_onset',
                    xEnd='event_offset',
                ),
                axis=1
            )

            event_onset = timing_table['event_onset'].values
            event_onset = (event_onset - 1) / freq
            event_offset = timing_table['event_offset'].values
            event_offset = (event_offset - 1) / freq
            event_duration = event_offset - event_onset

            trial_onset = event_onset[0]
            trial_offset = event_offset[-1]
            trial_duration = trial_offset - trial_onset
            trial_index = i + 1

            if stimulus_type == 'event':
                timing_table['event_onset'] = event_onset
                timing_table['event_offset'] = event_offset
                timing_table['event_duration'] = event_duration

                timing_table['trial_onset'] = trial_onset
                timing_table['trial_offset'] = trial_offset
                timing_table['trial_duration'] = trial_duration

                timing_table['trial_index'] = trial_index
                timing_table['event_index_in_trial'] = np.arange(len(timing_table)) + 1

                for col in event_table.index:
                    if col not in timing_table:
                        timing_table[col] = event_table[col]

                stimulus_data.append(timing_table)
            else:
                event_table['trial_onset'] = trial_onset
                event_table['trial_offset'] = trial_offset
                event_table['trial_duration'] = trial_duration

                event_table['trial_index'] = trial_index

                stimulus_data.append(event_table.to_frame().T)

        stimulus_data = pd.concat(stimulus_data, axis=0)
        if stimulus_type == 'event':
            stimulus_data['event_index'] = np.arange(len(stimulus_data)) + 1

    elif stimulus_type == 'session':
        recording_end = h5['for_preproc']['elec_data_raw'].shape[-1] / freq
        session_onset = (np.array(h5['for_preproc']['stitch_index_raw']) - 1) / freq
        if not len(session_onset.shape):
            session_onset = session_onset[..., None]
        session_offset = np.concatenate([session_onset[:-1], [recording_end]])
        session_index = np.arange(len(session_onset)) + 1
        stimulus_data = pd.DataFrame(dict(
            session_onset=session_onset,
            session_offset=session_offset,
            session_index=session_index
        ))

    else:
        raise ValueError('Unrecognized stimulus type %s' % stimulus_type)

    return stimulus_data


def get_word_table(event_table_path):
    stimulus_table = pd.read_csv(event_table_path)
    sel = stimulus_table.key.str.startswith('word')
    sel &= stimulus_table.event_duration < 5
    word_table = stimulus_table[sel]
    word_table = word_table.rename(
        dict(
            event_onset='onset',
            event_offset='offset',
            event_duration='duration',
        ),
        axis=1
    )
    return word_table


def get_fixation_table(event_table_path):
    stimulus_table = pd.read_csv(event_table_path)
    sel = stimulus_table.string.isin(('+', '[]')) | stimulus_table.key.isin(('preprobe', 'postprobe'))
    sel &= stimulus_table.event_duration < 5
    fixation_table = stimulus_table[sel]
    fixation_table = fixation_table.rename(
        dict(
            event_onset='onset',
            event_offset='offset',
            event_duration='duration',
        ),
        axis=1
    )
    return fixation_table


def get_langloc(h5):
    channel_ix = np.argsort(h5['elec_ch'])
    _channel_names = np.array(h5['elec_ch_label'])[channel_ix]
    _channel_names = remap_duplicate_channel_names(_channel_names)
    channel_names = _channel_names.tolist()
    n_sensors = len(channel_names)
    s_vs_n_sig = h5.get('s_vs_n_sig', None)
    if isinstance(s_vs_n_sig, dict):
        s_vs_n_sig = s_vs_n_sig['elec_data']
    else:
        s_vs_n_sig = np.full((n_sensors,), np.nan)
    s_vs_n_p_ratio = h5.get('s_vs_n_p_ratio', None)
    if isinstance(s_vs_n_p_ratio, dict):
        s_vs_n_p_ratio = s_vs_n_p_ratio['elec_data']
    else:
        s_vs_n_p_ratio = np.full((n_sensors,), np.nan)
    langloc_table = pd.DataFrame(dict(
        channel=channel_names,
        s_vs_n_sig=s_vs_n_sig,
        s_vs_n_p_ratio=s_vs_n_p_ratio
    ))
    
    return langloc_table


def save_stimulus_data(stimulus_data, output_path):
    stimulus_data.to_csv(output_path, index=False)


def remap_channel_type(x):
    if x.startswith('ecog'):
        return 'ecog'
    if x == 'ekg':
        return 'ecg'
    if x in ('ground', 'reference', 'empty'):
        return 'misc'
    return x


def remap_duplicate_channel_names(channel_names):
    prior_names = set()
    isarray = isinstance(channel_names, np.ndarray)
    channel_names = channel_names.tolist()
    for i in range(len(channel_names)):
        channel_name = str(channel_names[i])
        while channel_name in prior_names:
            channel_name = channel_name + '_'
        prior_names.add(channel_name)
        channel_names[i] = channel_name

    if isarray:
        channel_names = np.array(channel_names)

    return channel_names


def get_info(h5):
    # Channel indices
    channel_ix = np.argsort(h5['elec_ch'])

    # Channel names
    _channel_names = np.array(h5['elec_ch_label'])[channel_ix]
    _channel_names = remap_duplicate_channel_names(_channel_names)
    channel_names = _channel_names.tolist()

    # Channel types
    _channel_types = np.array(h5['elec_ch_type'])[channel_ix]
    for i in range(len(_channel_types)):
        _channel_types[i] = remap_channel_type(_channel_types[i])
    channel_types = _channel_types.tolist()

    # Bad channels
    goods = set(h5['elec_ch_clean']) & set(np.where(h5['elec_ch_valid'])[0].tolist())
    bads = set(h5['elec_ch']) - goods
    for x in ('elec_ch_prelim_deselect', 'elec_ch_user_deselect', 'elec_ch_with_IED'):
        if x in h5:
            arr = np.array(h5[x])
            if not len(arr.shape):
                arr = arr[..., None]
            arr = arr.tolist()
            bads |= set(arr)
    misc = channel_ix[_channel_types == 'misc']
    for bad in misc:
        bads.add(bad)
    bads = np.sort(np.array([x for x in bads], dtype=int) - 1)
    bads = _channel_names[bads].tolist()

    # Sampling frequency
    sfreq = h5['for_preproc']['sample_freq_raw']

    # Info object
    info = mne.create_info(
        channel_names,
        sfreq,
        ch_types=channel_types
    )
    info['bads'].extend(bads)

    return info


def get_raw(h5):
    info = get_info(h5)
    raw = h5['for_preproc']['elec_data_raw']
    raw = mne.io.RawArray(raw, info)

    return raw


def save_raw(raw, output_path):
    raw.save(output_path, overwrite=True)


def get_channel_masks(langloc_dir, overwrite=False):
    mask_dir = join(dirname(dirname(__file__)), 'resources', 'masks')
    suffix_in = '_langloc.csv'
    suffix_out = '_channel_mask.csv'
    keyword = '_MITLangloc'
    if os.path.exists(langloc_dir):
        for filename in [x for x in os.listdir(langloc_dir) if (keyword in x and x.endswith(suffix_in))]:
            subject = filename[:-len(suffix_in)].replace(keyword, '')
            path = join(langloc_dir, filename)
            channel_mask = pd.read_csv(path)[['channel', 's_vs_n_sig']].rename(dict(s_vs_n_sig='include'), axis=1)
            if not os.path.exists(mask_dir):
                os.makedirs(mask_dir)
            out_path = join(mask_dir, subject + suffix_out)
            if overwrite or not os.path.exists(out_path):
                channel_mask.to_csv(out_path, index=False)
    else:
        stderr('langloc_dir %s does not exist, skipping channel mask extraction' % langloc_dir)



if __name__ == '__main__':
    argparser = argparse.ArgumentParser(textwrap.dedent('''\
        Convert MATLAB objects from EvLab ECoG pipeline to HDF5 (brain data) and CSV (stimuli).'''
    ))
    argparser.add_argument('input_paths', nargs='+', help='Input directory containing MATLAB (*.mat) objects.')
    argparser.add_argument('output_dir', help='Output directory in which to place HDF5 and CSV files.')
    argparser.add_argument('-f', '--force_restart', action='store_true',
                           help='Force restart. Otherwise, skip existing files.')
    args = argparser.parse_args()

    input_paths = args.input_paths
    output_dir = args.output_dir
    force_restart = args.force_restart

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mfile_path = join(dirname(dirname(__file__)), 'resources', 'matlab', 'get_ecog.m')

    input_path = None
    for input_path in args.input_paths:
        stderr('Processing file %s\n' % input_path)
        name = basename(input_path)[:-len(SUFFIX)]
        output_path_base = join(output_dir, name)

        deps = [[input_path, mfile_path]]

        h5_path = output_path_base + '.h5'
        h5_mtime, h5_exists = check_deps(h5_path, deps)
        h5_stale = h5_mtime == 1
        do_h5 = force_restart or (not h5_exists or h5_stale)
        if do_h5:
            initialize()
            matlab_dir = os.path.abspath(join(dirname(dirname(__file__)), 'resources', 'matlab'))
            for matlab in ('matlab', 'matlab.exe'):
                failure = os.system('which %s' % matlab)
                if not failure:
                    break
            assert not failure, 'MATLAB executable not found'

            stderr('  Exporting from MATLAB\n')
            stderr(MATLAB_CMD % (matlab, matlab_dir, input_path, h5_path) + '\n')
            failure = os.system(MATLAB_CMD % (matlab, matlab_dir, input_path, h5_path))
            assert not failure, 'Failure on file %s' % input_path

        deps.append(h5_path)

        session_path = output_path_base + '_stim_session.csv'
        session_mtime, session_exists = check_deps(session_path, deps)
        session_stale = session_mtime == 1
        do_session = force_restart or (not session_exists or session_stale)

        trial_path = output_path_base + '_stim_trial.csv'
        trial_mtime, trial_exists = check_deps(trial_path, deps)
        trial_stale = trial_mtime == 1
        do_trial = force_restart or (not trial_exists or trial_stale)

        event_path = output_path_base + '_stim_event.csv'
        event_mtime, event_exists = check_deps(event_path, deps)
        event_stale = event_mtime == 1
        do_event = force_restart or (not event_exists or event_stale)

        langloc_path = output_path_base + '_langloc.csv'
        langloc_mtime, langloc_exists = check_deps(langloc_path, deps)
        langloc_stale = langloc_mtime == 1
        do_langloc = force_restart or (not langloc_exists or langloc_stale)

        fif_path = output_path_base + '_ieeg.fif'
        fif_mtime, fif_exists = check_deps(fif_path, deps)
        fif_stale = fif_mtime == 1
        do_fif = force_restart or (not fif_exists or fif_stale)

        if do_session or do_trial or do_event or do_langloc or do_fif:
            # Get MATLAB data
            stderr('  Loading from MATLAB\n')
            h5 = get_matlab_data(h5_path)

            # Get and save session data
            if do_session:
                stderr('  Saving session table\n')
                session_table = get_stimulus_data(h5, stimulus_type='session')
                save_stimulus_data(session_table, session_path)

            # Get and save trial data
            if do_trial:
                stderr('  Saving trial table\n')
                trial_table = get_stimulus_data(h5, stimulus_type='trial')
                save_stimulus_data(trial_table, trial_path)

            # Get and save event data
            if do_event:
                stderr('  Saving event table\n')
                event_table = get_stimulus_data(h5)
                save_stimulus_data(event_table, event_path)

            # Get and save langloc data
            if do_langloc:
                stderr('  Saving LangLoc table\n')
                langloc_table = get_langloc(h5)
                save_stimulus_data(langloc_table, langloc_path)

            # Get and save raw signal
            if do_fif:
                stderr('  Saving signals (FIF)\n')
                raw = get_raw(h5)
                save_raw(raw, fif_path)

        word_path = output_path_base + '_stim_word.csv'
        word_mtime, word_exists = check_deps(word_path, deps + [event_path])
        word_stale = word_mtime == 1
        do_word = force_restart or (not word_exists or word_stale)
        if do_word:
            word_table = get_word_table(event_path)
            save_stimulus_data(word_table, word_path)

        fixation_path = output_path_base + '_stim_fixation.csv'
        fixation_mtime, fixation_exists = check_deps(fixation_path, deps + [event_path])
        fixation_stale = fixation_mtime == 1
        do_fixation = force_restart or (not fixation_exists or fixation_stale)
        if do_fixation:
            fixation_table = get_fixation_table(event_path)
            save_stimulus_data(fixation_table, fixation_path)

    if input_path is not None:
        get_channel_masks(output_dir, overwrite=force_restart)
