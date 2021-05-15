#!/usr/bin/env python

from pip._internal import main as pip
pip(['install', '--upgrade', 'pip'])
pip(['install', '--user', 'pandas'])
pip(['install', '--user', 'numpy'])
pip(['install', '--user', 'sklearn'])
pip(['install', '--user', 'catboost'])
pip(['install', '--user', 'pyarrow'])
