#!/bin/bash
pip uninstall twisterl -y
pip install .
python3 -m twisterl.train --config /Users/jona/Documents/GIT/RESEARCH/twisteRL/examples/ppo_puzzle8_v1.json