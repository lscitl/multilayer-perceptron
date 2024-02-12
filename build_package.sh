#!/bin/sh

source $PWD/bin/activate

python -m build ./smlp_package

pip install ./smlp_package/dist/smlp-0.0.1-py3-none-any.whl