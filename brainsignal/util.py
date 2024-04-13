import sys
import os

from brainsignal.constants import *


def stderr(s):
    sys.stderr.write(s)
    sys.stderr.flush()


def join(*args):
    args = [os.path.normpath(x) for x in args]

    return os.path.normpath(os.path.join(*args))


def basename(path):
    path = os.path.normpath(path)

    return os.path.basename(path)


def dirname(path):
    path = os.path.normpath(path)

    return os.path.dirname(path)


def exists(path):
    path = os.path.normpath(path)

    return os.path.exists(path)


def get_path(output_dir, path_type, action_type, action_id, subject=None):
    assert action_type in PATHS, 'Unrecognized action_type %s' % action_type
    assert path_type in PATHS[action_type], 'Unrecognized path_type %s for action_type %s' % (path_type, action_type)
    path = PATHS[action_type][path_type]

    if '%s' in path and subject is not None:
        path = path % subject

    if path_type == 'subdir':
        suffix = join(path, action_id)
    else:
        prefix = PATHS[action_type]['subdir']
        suffix = join(prefix, action_id, path)

    path = join(output_dir, suffix)

    return path


def infer_stimulus_table_path_from_raw(
        stimulus_type=None,
        raw_path=None
):
    assert raw_path is not None, 'Either a stimulus path or a FIF path must be provided'
    suffix = get_fif_suffix(raw_path)
    if stimulus_type is None:
        stimulus_type = 'event'
    path_base = raw_path[:len(raw_path) - len(suffix)]
    stimulus_table_path = path_base + f'_stim_{stimulus_type}.csv'

    return stimulus_table_path


def getmtime(path):
    path = os.path.normpath(path)
    if exists(path):
        return os.path.getmtime(path)

    return None


def get_max_mtime(*mtimes):
    mtimes = [x for x in mtimes if x is not None]
    if mtimes:
        return max(mtimes)

    return None


def is_stale(target_mtime, dep_mtime):
    return target_mtime and dep_mtime and target_mtime < dep_mtime


def check_deps(path, dep_seq):
    mtime = getmtime(path)
    _exists = mtime is not None

    if dep_seq:
        deps = dep_seq[-1]
        if not isinstance(deps, list):
            deps = [deps]
        _dep_seq = dep_seq[:-1]
        dep_mtime = None
        for dep in deps:
            _dep_mtime, _ = check_deps(dep, _dep_seq)
            if _dep_mtime == 1:
                return 1, _exists
            if is_stale(mtime, _dep_mtime):
                return 1, _exists
            dep_mtime = get_max_mtime(dep_mtime, _dep_mtime)

        mtime = get_max_mtime(mtime, dep_mtime)

    return mtime, _exists


RESOURCE_DIR = join(dirname(__file__), 'resources')


def get_overwrite(overwrite):
    if isinstance(overwrite, dict):
        return overwrite

    out = dict(
        preprocess=False,
        epoch=False,
        plot=False,
    )
    if overwrite is None:
        for x in out:
            out[x] = True
    elif isinstance(overwrite, str):
        assert overwrite in out, 'Unrecognized value for overwrite: %s' % overwrite
        out[overwrite] = True
    elif overwrite is False:
        pass
    else:
        raise ValueError('Unrecognized value for overwrite: %s' % overwrite)

    return out


def get_fif_suffix(path):
    suffix = ''
    for _suffix in MNE_SUFFIXES:
        if path.endswith(_suffix):
            return _suffix

    return suffix