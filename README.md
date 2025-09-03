# Autol_RoboSense_track1
## phase1
Phase 1 uses the open source model DriveLMMo1.
The download link is https://huggingface.co/ayeshaishaq/DriveLMMo1.
  
It should be noted that when loading the model for the first time, a problem of "mixed use of tabs and spaces for indentation" may be reported in a Python file, need to locate and fix it yourself.  

The environment configuration file exported by anaconda is environment.yml, where torch==2.7.1 and transformers==4.51.3.

After placing the downloaded model in the appropriate location, modify the specified path in the phase1_inference.py file to start inference.
