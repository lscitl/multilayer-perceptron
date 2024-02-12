#!/bin/sh

source $PWD/bin/activate

if [ ! -e "./smlp_package/dist" ]; then
    python -m build ./smlp_package
fi

pip install ./smlp_package/dist/smlp-*.whl