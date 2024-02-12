#!/bin/sh

brew install python@3.10

if [ ! -e ./bin/activate ]; then
    if [ ! -e ~/goinfre/.brew/bin/python3.10 ]; then
        python3.10 -m venv ./
    else
        ~/goinfre/.brew/bin/python3.10 -m venv ./
    fi
fi

source ./bin/activate

pip install --upgrade pip

pip install numpy pandas matplotlib black seaborn tensorflow ./src