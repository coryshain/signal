import sys
import os


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


def getmtime(path):
    path = os.path.normpath(path)

    return os.path.getmtime(path)


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
