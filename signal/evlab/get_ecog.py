import textwrap
import argparse
from pymatreader import read_mat
import numpy as np
import pandas as pd
import mne

from signal.util import *
from signal.evlab.initialize import initialize


SUFFIX = '_crunched.mat'
MATLAB_CMD = textwrap.dedent('''\
    %s -nodisplay -nosplash -nodesktop -wait -r "addpath(\\"%s\\"); get_ecog(\\"%s\\", \\"%s\\"); exit;"\
''')


def get_matlab_data(path):
    return read_mat(path)['s']


def get_stimuli(h5):
    events_table = pd.DataFrame(h5['events_table'])

    return events_table


def get_langloc(h5):
    channel_ix = np.argsort(h5['channel_indices'])
    channel_names = np.array(h5['channel_names'])[channel_ix]
    n_sensors = len(channel_names)
    s_vs_n_sig = h5.get('s_vs_n_sig', np.nan)
    if isinstance(s_vs_n_sig, dict):
        s_vs_n_sig = s_vs_n_sig['elec_data']
    else:
        s_vs_n_sig = np.full((n_sensors,), np.nan)
    s_vs_n_p_ratio = h5.get('s_vs_n_p_ratio', np.nan)
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


def save_stimuli(events_table, output_path):
    events_table.to_csv(output_path, index=False)


def remap_channel_type(x):
    if x.startswith('ecog'):
        return 'ecog'
    if x == 'ground':
        return 'misc'
    if x == 'ekg':
        return 'ecg'
    return x


def get_info(h5):
    # Channel indices
    channel_ix = np.argsort(h5['channel_indices'])

    # Channel names
    _channel_names = np.array(h5['channel_names'])[channel_ix]
    channel_names = _channel_names.tolist()

    # Channel types
    _channel_types = np.array(h5['channel_types'])[channel_ix]
    for i in range(len(_channel_types)):
        _channel_types[i] = remap_channel_type(_channel_types[i])
    channel_types = _channel_types.tolist()

    # Bad channels
    bads = []
    for x in h5['bad_channels']:
        _bads = np.array(h5['bad_channels'][x] - 1, dtype=int)
        if not _bads.shape:
            _bads = _bads[..., None]
        bads.append(_bads)
    bads.append(channel_ix[_channel_types == 'misc'])
    bads = np.concatenate(bads)
    bads = _channel_names[bads].tolist()

    # Sampling frequency
    sfreq = h5['freq']

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
    raw = h5['raw']
    raw = mne.io.RawArray(raw, info)

    return raw


def save_raw(raw, output_path):
    raw.save(output_path, overwrite=True)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(textwrap.dedent('''\
        Convert MATLAB objects from EvLab ECoG pipeline to HDF5 (brain data) and CSV (stimuli).'''
    ))
    argparser.add_argument('input_dir', help='Input directory containing MATLAB (*.mat) objects.')
    argparser.add_argument('output_dir', help='Output directory in which to place HDF5 and CSV files.')
    argparser.add_argument('-f', '--force_restart', action='store_true',
                           help='Force restart. Otherwise, skip existing files.')
    args = argparser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    force_restart = args.force_restart

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for input_path in [x for x in os.listdir(input_dir) if x.endswith(SUFFIX)]:
        stderr('Processing file %s\n' % input_path)
        name = input_path[:-len(SUFFIX)]
        input_path = join(input_dir, input_path)
        output_path_base = join(output_dir, name)

        h5_path = output_path_base + '.h5'
        h5_mtime, h5_exists = check_deps(h5_path, [input_path])
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
            failure = os.system(MATLAB_CMD % (matlab, matlab_dir, input_path, h5_path))
            assert not failure, 'Failure on file %s' % input_path

        stim_path = output_path_base + '_stim.csv'
        stim_mtime, stim_exists = check_deps(stim_path, [input_path, h5_path])
        stim_stale = stim_mtime == 1
        do_stim = force_restart or (not stim_exists or stim_stale)

        langloc_path = output_path_base + '_langloc.csv'
        langloc_mtime, langloc_exists = check_deps(langloc_path, [input_path, h5_path])
        langloc_stale = langloc_mtime == 1
        do_langloc = force_restart or (not langloc_exists or langloc_stale)

        fif_path = output_path_base + '_ieeg.fif'
        fif_mtime, fif_exists = check_deps(fif_path, [input_path, h5_path])
        fif_stale = fif_mtime == 1
        do_fif = force_restart or (not fif_exists or fif_stale)

        if do_stim or do_fif or do_langloc:
            # Get MATLAB data
            stderr('  Loading from MATLAB\n')
            h5 = get_matlab_data(h5_path)

            # Get and save stimulus data
            if do_stim:
                stderr('  Saving stimulus table\n')
                events_table = get_stimuli(h5)
                save_stimuli(events_table, stim_path)

            # Get and save langloc data
            if do_langloc:
                stderr('  Saving LangLoc table\n')
                langloc_table = get_langloc(h5)
                save_stimuli(langloc_table, langloc_path)

            # Get and save raw signal
            if do_fif:
                stderr('  Saving signals (FIF)\n')
                raw = get_raw(h5)
                save_raw(raw, fif_path)