#!/bin/sh

if ! command -v python3.10 &> /dev/null; then
    echo "python 3.10 is not found. Install python 3.10"
    brew install python@3.10
fi

# setup venv
echo "setup virtual environment."
if [ ! -e ./bin/activate ]; then
    if [ ! -e ~/goinfre/.brew/bin/python3.10 ]; then
        python3.10 -m venv ./
    else
        ~/goinfre/.brew/bin/python3.10 -m venv ./
    fi
fi

source ./bin/activate

echo "upgrade pip and install packages..."
pip -q install --upgrade pip
pip -q install numpy pandas matplotlib black seaborn tensorflow build

echo "environment setting is finished!"
echo "run 'source ./bin/activate' to activate virtual environment."