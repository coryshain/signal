import os

from brainsignal.util import *

LIB_ECOG_REPO_NAMES = [
    'ecog_pipeline_merged',
    'ecog_MITLangloc',
    'ecog_MITNaturalisticStoriesTask',
    'ecog_MITConstituentBounds',
]
GITHUB_URL = 'git@github.mit.edu:ccasto/%s.git'


def initialize():
    for lib in LIB_ECOG_REPO_NAMES:
        matlabdir = join(RESOURCE_DIR, 'matlab')
        if not exists(matlabdir):
            os.makedirs(matlabdir)
        repo_path = join(matlabdir, lib)
        if not exists(repo_path):
            failure = os.system('git clone %s %s' % (GITHUB_URL % lib, repo_path))
            assert not failure, 'Installation failed for repo %s' % (GITHUB_URL % lib)



if __name__ == '__main__':
    initialize()
