#!/usr/bin/env bash
set -e
python3 -m pip install -r requirements.txt
python3 run_pipeline.py
python3 app.py
