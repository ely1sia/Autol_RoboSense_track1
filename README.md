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
During this phase, we fine-tuned the Internvl3-9b model in two stages. In the first stage, we used approximately 20,000 data points from the Drivelmmo1 dataset to fine-tune the model, enabling it to perform inference before outputting the final result. In the second stage, we combined the Drive_bench dataset and a dataset of partially damaged images, yielding approximately 1,354 data points. We then used Kimi to expand this dataset with a chain of thought, supplementing the reasoning process within the dataset. Afterwards, we integrated the first-stage fine-tuning data with the second-stage fine-tuning data in a ratio of 2:8, and fine-tuned the Internvl3-9b model fine-tuned in the first stage a second time to obtain the final model.  

result_phase2.json is the result file we finally submit to the leaderboard.  
Considering the Internvl pre-training specifications, we wrote some coordinate transformation, image stitching and other contents in the phase2_inference.py file. Similarly, the same work was done for the fine-tuning data.  



