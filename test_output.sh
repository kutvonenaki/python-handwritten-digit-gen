#!/bin/bash

#check if the nosetest command exists
command -v nosetests &> /dev/null
if [ $? -ne 0 ]; then
    echo 'python-nose package required. Load with pip install nose, or alternatively run the test_sample.py by hand'
    exit 1
fi

#run the tests inside /tests/
nosetests --nocapture --no-skip --verbose tests/*.py
