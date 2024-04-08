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
        kwargs='preprocess_kwargs.yml',
        subdir=ACTION_VERB_TO_NOUN['preprocess'],
        output='%%s%s' % RAW_SUFFIX,
    ),
    epoch=dict(
        kwargs='epoch_kwargs.yml',
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