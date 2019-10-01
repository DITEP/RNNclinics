# Phase 1 Screening Prediction Web Tool
Authors: [Hichem Aloui](https://github.com/hichem3), Loic Verlingue, Ugo Benassayag

## What is it?
We have develloped a Hierarchical Attention Model (HAN) to predcit the inclusion of patient in phase one clinical trials in oncology. That is, to predict whether the patient will pass screening and DLT (Dose Limiting Toxicity).
The paper is under submission, the code is open-source in this repo.

This user-friendly web interface can help doctors use this model to perform predictions.

## How it works? 
This repo is under development.

Currently it needs python IDE with Keras/Tensorflow installed and loaded HAN model to run the Flasktest.py to open the interface.
We are working on a more straightforward approche for the user.

The current login/password are admin/password. Once entered, user access the index page.

The report of the inclusion consultation of the pateint can be loaded or written in web page (index.html). 
Flask app ( index.py ) calls python code ( pred.py ) that analyses the input (the medical report) and returns a probability of inclusion (Succesful Screening and DLT period completion) based on our pre-trained HAN. It also returns the attention value of each sentence and display them to the user. Attention value of a sentence can be perceived as the weight of the sentence in the whole document, how important it is in the algorithm decision making. Sentences with the highest attention values are highlighted to show the user the most significant part of the report.


## Future work

-Link to a cluster for faster computation and getting rid of Python Modules necessity

-Link to a database

-Host on a server with restrected access

-Evaluate prospectively in a clinical trial. Partners are welcomed to contact us at loic.verlingue@gustaveroussy.fr 


## Contribution
Contributions and user feebdacks are very welcomed.

Contact: loic.verlingue@gustaveroussy.fr 
