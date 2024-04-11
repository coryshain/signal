CFG_FILENAME = 'config.yml'
RAW_SUFFIX = '-raw.fif.gz'
EPOCHS_SUFFIX = '-epo.fif.gz'
PLOT_SUFFIX = '_evoked.png'
MNE_SUFFIXES = [
    '-raw.fif',
    '_ieeg.fif',
    '_eeg.fif',
    '_meg.fif',
    '-epo.fif'
]
MNE_SUFFIXES += [x + '.gz' for x in MNE_SUFFIXES]
ACTION_VERB_TO_NOUN = dict(
    preprocess='preprocessing',
    epoch='epoching',
    plot='plotting',
)
PATHS = dict(
    preprocess=dict(
        kwargs='%s_preprocess_kwargs.yml',
        subdir=ACTION_VERB_TO_NOUN['preprocess'],
        output='%%s%s' % RAW_SUFFIX,
    ),
    epoch=dict(
        kwargs='%s_epoch_kwargs.yml',
        subdir=ACTION_VERB_TO_NOUN['epoch'],
        output='%%s%s' % EPOCHS_SUFFIX,
    ),
    plot=dict(
        kwargs='plot_kwargs.yml',
        subdir=ACTION_VERB_TO_NOUN['plot'],
        image='%%s%s' % PLOT_SUFFIX,
        output='%s.csv' % ACTION_VERB_TO_NOUN['plot'],
    )
)
DEPS = dict(
    preprocess=None,
    epoch='preprocess',
    plot='epoch'
)
DEPS_REV = {DEPS[x]: x for x in DEPS}


def get_dep_seq_from_dep_map():
    key = None
    _DEPS_REV = DEPS_REV.copy()
    seq = []
    while len(_DEPS_REV):
        key = _DEPS_REV.pop(key)
        seq.append(key)

    return seq


DEP_SEQ = get_dep_seq_from_dep_map()
DEP_SEQ_REV = DEP_SEQ[::-1]
