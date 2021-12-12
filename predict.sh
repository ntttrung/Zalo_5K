#!/bin/bash
python3 detect.py --weights .saved_model/full_data.pt --source data/images --conf 0.4 --save-txt