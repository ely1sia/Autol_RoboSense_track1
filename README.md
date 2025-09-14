# Autol_RoboSense_track1
## Phase1
Phase 1 uses the open source model DriveLMMo1.
The download link is https://huggingface.co/ayeshaishaq/DriveLMMo1.
  
It should be noted that when loading the model for the first time, a problem of "mixed use of tabs and spaces for indentation" may be reported in a Python file, need to locate and fix it yourself.  

The environment configuration file exported by anaconda is environment.yml, where torch==2.7.1 and transformers==4.51.3.

After placing the downloaded model in the appropriate location, modify the specified path in the phase1_inference.py file to start inference.  

Finally, for the result json file obtained by the inference script, execute data_process.py to remove the reasoning steps in the answers of some perception-type QA pairs and only keep the final results, that is, to obtain the final result file.  

The results.json file in the folder is the result file that our team submitted to the leaderboard in Phase 1.

## Phase2


