#!/bin/sh

source $PWD/bin/activate

rm -rf "./smlp_package/dist"

python -m build ./smlp_package

pip install ./smlp_package/dist/smlp-*.whl