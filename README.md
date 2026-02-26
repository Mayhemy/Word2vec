

## Word2vec Skip-gram implementation

### Goal of the project
The goal of this project is to implement Skip-gram in several different ways from scratch.

**Naive Approach** - implemented fully and tested. In this approach library numpy is used to implement forward pass, loss, gradients, and parameter updates.
**Optimized Approach** - to be implemented. Plan is to implement negative sampling and vectorization, along with some cool tricks I picked up while implementing it Naively.
**Comparison** - this mode will just compare the execution times of optimized and naive approach.


### How to run
Linux has a Makefile while Windows has a .bat file

**Linux / macOS** :  *Makefile* is provided for simple execution. 
***Prerequisite***: Python3, along with venv module needs to be installed on the host environment (the environment from which you run the script) for the script to work : ```sudo apt install python3-venv```  on Ubuntu. 
Every time a program is run via Makefile,  the setup is executed, which creates a virtual environment and installs the requirements and then runs the code based on the arguments provided, the options are:

 - ```make run-naive``` (runs Naive Approach)
 - ```make run-optimized``` (runs Optimized Approach - not yet implemented)
 - ```make run-compare``` (runs both and compares them - not yet implemented)

Utility function ```make clean``` is added in order to delete the venv folder and delete pycache.

**Windows** : Batch script *run.bat* is provided for simple execution.
***Prerequisite***: Python3 needs to be installed on the host environment (the environment from which you run the script)
Run the script by double-clicking the *run.bat* file or use PowerShell or cmd using the following command:

 - ``` .\run.bat``` (runs Naive Approach)

 Nothing apart from Naive Approach is supported for the Windows script, since nothing apart from that is implemented yet.
 
Utility batch file ```.\clean.bat``` is added in order to delete the venv folder and delete pycache.
