

## Word2vec Skip-gram implementation

### Goal of the project
The goal of this project is to implement Skip-gram in several different ways from scratch.

**Naive Approach** - In this approach I used mainly for loops and a little bit of numpy to implement forward pass, loss, gradients, and parameter updates.

**Optimized Approach** -Implemented negative sampling and vectorization fully using Numpy along with forward pass, loss, gradients, and parameter updates.

**Comparison** - this mode will run and compare Optimized and Naive approach.


### How to run
Linux has a Makefile while Windows has a .bat file

**Linux / macOS** :  *Makefile* is provided for simple execution. 
***Prerequisite***: Python3, along with venv module needs to be installed on the host environment (the environment from which you run the script) for the script to work : ```sudo apt install python3-venv```  on Ubuntu. 
The first time you run Makefile, setup is executed, which creates a virtual environment and installs the requirements and then runs the code based on the arguments provided, the options are:

 - ```make run-naive``` (runs Naive Approach)
 - ```make run-optimized``` (runs Optimized Approach)
 - ```make run``` and ```make run-compare``` (runs both approaches and compares them)

Utility function ```make clean``` is added in order to delete the venv folder and delete pycache.

**Windows** : Batch scripts are provided for simple execution.
***Prerequisite***: Python3 needs to be installed on the host environment (the environment from which you run the script)
Run the script by double-clicking the *run.bat* file or use PowerShell or cmd using the following command:

 - ``` .\run.bat``` (runs Comparison mode)
 - ``` .\run_optimized.bat``` (runs Optimized approach)
 
Utility batch file ```.\clean.bat``` is added in order to delete the venv folder and delete pycache.
