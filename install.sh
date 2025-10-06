#!/bin/bash
pip install -r requirements.txt
MAX_JOBS=4 pip install flash-attn --no-build-isolation
