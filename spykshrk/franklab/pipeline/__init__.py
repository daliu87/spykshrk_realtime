import json
import os
import datajoint as dj
from . import nspike_schema
from . import exceptions

root_path = os.path.abspath(os.path.join(globals()['__path__'][0], '../../../'))


local_cred_filename = os.path.join(root_path, 'datajoint/local_cred.ini')
with open(local_cred_filename) as f:
    local_cred = json.load(f)

cred_filename = "~/"

dj.config['database.host'] = local_cred['host']
dj.config['database.user'] = local_cred['user']
dj.config['database.password'] = local_cred['password']
