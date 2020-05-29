#!/usr/bin/env bash

bin_path=$(command -v blaze)

if [[ ${bin_path} == '' ]]; then
    echo "No blaze -- running with the ordinary Python binary"

    python test.py
else
    echo "Found blaze -- running with the Blaze binary"

    blaze run -c opt \
        test \
        --
fi
