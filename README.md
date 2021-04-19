# covid-death-prediction
Corona Virus Death Prediction


Authors: Mitchell Driedger, Laroy Milton, Mohammad Mosleh

#### Setup
Use the command `source setup_env.sh` to create a python virtual environment, install numpy, matplotlib, pandas, sklearn, seaborn,
and then activate the virtual environment.   
Use `deactivate` to exit the active virtual environment if needed.

#### Usage 
+ Steps to run `Project3.py` or `mp.py` are:
    - Ensure `source setup_env.sh` has been run at least once previously 
    - Run `source activate_env.sh` to activate the virtual python environment 
    - Run `python3 Project3.py` to run the main project file
    - Run `python3 mp.py` to run the Multi-layer Perceptron project file

+ Use `deactivate` to exit the active virtual environment if needed.

#### Directories and files
- `./DataFiles/`
  + `Covid-60weeks.csv`
  + `Health.csv`
  + `Sanitation.csv`
  + `Tourism.csv`
  + `Fitness.csv`
  + `Economics.csv`
  + `Demographics.csv`
- `./Plots/`: Created at the end of the virtual environment setup. This directory will contain all the plots from the 
  data visualizations and model results.
- `Project3.py`: The main project file. Running it will initially create 6 graphs for data visualization. Next it will 
  begin going through each model and its variations of features. It will show the progress by printing the current model
  being trained. Expected run time is around 25 minutes.
- `mp.py`: Multi-layer Perceptron model run file. Outputted graphs will be in Plots.
- `setup_env.sh`: contains the script to setup and run the virtual python environment needed to install and use the 
  required python packages.
- `activate_env.sh`: Used to activate the existing virtual python environment.
- `README.md`: This file
