# cptone_ds
capital one data science challenge

To set up your environment to run this code follow these steps:


1- Create a virtual environment
   python3 -m venv ./.my_env
   
2- Activate the virtual environment (if still in current directory)
   source ./.my_env/bin/activate

3- use pip to install all the requirements of the environment from requirements.txt file
   pip install -r requirements.txt
   
   
***How to add virtual env to jupyter notebook***

- make sure the environment is activated
pip install --user ipykernel

- virtual env to jupyter 
python -m ipykernel install --user --name=.my_env



***Before running the code a folder called data needs to be created in the directory
that the code exists. There is a parser function in fraud_detector.py that should 
be able to download the data zip file from public github link and download it.***