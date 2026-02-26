@echo off
if not exist venv (
    py -m venv venv
    venv\Scripts\pip install numpy==1.26.4
)
venv\Scripts\python word2vec.py -m naive
pause