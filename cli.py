#!/usr/bin/env python3
# Top-level CLI that delegates to the package code.
# Use absolute import so it works when run as `python cli.py`.

from eegcfct.train.runner import main

if __name__ == "__main__":
    main()
