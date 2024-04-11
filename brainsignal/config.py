import os
import yaml
import copy

from brainsignal.constants import *


def get_config(path):
    out = {}
    if os.path.exists(path):
        with open(path, 'r') as f:
            out = yaml.safe_load(f)

    return out


def get_data_info(cfg):
    load_kwargs = []
    for _load_kwargs in cfg.get('data'):
        _load_kwargs['drop_bads'] = cfg.get('drop_bads', False)
        _load_kwargs['stimulus_type'] = cfg.get('stimulus_type', None)
        _load_kwargs['channel_mask_path'] = _load_kwargs.get('channel_mask_path', cfg.get('channel_mask_path', None))
        _load_kwargs['event_duration'] = _load_kwargs.get('event_duration', None)
        load_kwargs.append(_load_kwargs)

    return load_kwargs


def get_kwargs(cfg, action_type, action_id):
    if action_id is not None and action_type in cfg:
        kwargs = copy.deepcopy(cfg[action_type][action_id])
        if kwargs is None:
            kwargs = {}
        kwargs.update({'%s_id' % ACTION_VERB_TO_NOUN[action_type]: action_id})
    else:
        kwargs = None

    return kwargs


def get_action_sequence(
        cfg,
        action_type=None,
        action_id=None,
        action_sequence=None
):
    if action_type is None:
        action_type = 'plot'
    if action_type not in cfg:
        if action_type == 'preprocess':
            raise ValueError('No preprocessing instructions found in config')
        return get_action_sequence(
            cfg,
            action_type=DEPS[action_type],
            action_id=action_id,
            action_sequence=action_sequence
        )
    if action_id is None:
        action_id = list(cfg[action_type].keys())[0]
    if action_sequence is None:
        action_sequence = []
    assert action_id in cfg[action_type], 'No entry %s found in %s' % (action_id, action_type)
    kwargs = get_kwargs(cfg, action_type, action_id)
    action = dict(
        type=action_type,
        id=action_id,
        kwargs=kwargs
    )
    if len(action_sequence):
        action_sequence[0]['kwargs']['%s_id' % ACTION_VERB_TO_NOUN[action_type]] = action_id
    action_sequence.insert(0, action)

    dep_action_type = DEPS[action_type]
    if dep_action_type is None:
        return action_sequence

    dep_action_id = cfg[action_type][action_id].get('%s_id' % ACTION_VERB_TO_NOUN[dep_action_type], None)
    return get_action_sequence(
        cfg,
        action_type=dep_action_type,
        action_id=dep_action_id,
        action_sequence=action_sequence
    )

