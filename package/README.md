# Setting up Conda environment

###### Procedure Notes
* Usage assumes an active Google Earth Engine account. [Signup Here](https://signup.earthengine.google.com/#!/). Note that it may take several days for Google to process.
* Usage assumes installation of Conda.
* Usage assumes Conda environment config file `irr30.yml`downloaded to local computer in same folder as command prompt.
* Intent is for install on local computer.
* Procedure validated by running [notebook](https://github.com/sonalthakkarBerkeley/MIDS_Capstone_Summer2020/blob/master/dev/pipeline/Package_Example.ipynb) on the following platforms/command prompts:
    1. MacOs Catalina 10.15.5 Terminal
    2. Windows 10 Enterprise Anaconda Prompt
* Reference [Google Earth Engine Conda instructions](https://developers.google.com/earth-engine/python_install-conda) for troubleshooting.
* If issues activating environment, attempt `conda init <SHELL>` where `<SHELL>` is replaced by `bash` or `zsh` as applicable to your setup.
* If issues running `earthengine authenticate`, VPN may be restricting necessary connections.

###### Importing & Activating Environment
1. `conda env create -f irr30.yml`
2. `conda activate irr30`

###### First Usage
Assuming an active Google Earth Engine account, once in environment, authenticate with Google Earth Engine with the following command. Follow prompts.
`earthengine authenticate`

###### Loading Jupyter
`jupyter notebook`









#######################
Alternative install

INSTALL INSTRUCTIONS WITH VIRTUALENV
1. pip3 install virtualenv
2. virtualenv env
3. source env/bin/activate
4. curl https://bootstrap.pypa.io/get-pip.py | python3
5. pip3 install -r requirements.txt
6. jupyter notebook 

SHUTDOWN:
1. deactivate
