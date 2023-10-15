#!/bin/sh

if [ ! -e ~/goinfre/.brew/bin/python3.10 ]; then
    brew install python@3.10
fi

if [ ! -e ./bin/activate ]; then
    ~/goinfre/.brew/bin/python3.10 -m venv ./
fi

source ./bin/activate

pip install numpy pandas matplotlib black seaborn tensorflow