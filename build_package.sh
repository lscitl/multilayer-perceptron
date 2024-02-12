#!/bin/sh

source $PWD/bin/activate

python -m build ./smlp_package

pip install ./smlp_package/dist/smlp-*.whl