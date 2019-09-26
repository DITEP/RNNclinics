# Phase 1 Screening Prediction Web Tool
Authors: [Hichem Aloui](https://github.com/hichem3), Loic Verlingue

## What is it?
We have develloped an HAN model to predcit the inclusion of patient in phase one clinical trials in oncology.
The paper is under submission, the code is open-source in this repo.

This user-friendly web interface can help doctors use this model to perform predictions.

## How it works? 
This repo is under devellopment.

Currently it needs python IDE with Keras/Tensorflow installed and loaded HAN model to run the Flasktest.py to open the interface.
We are working on a more straightforward approche for the user.

The report of the inclusion consultation of the pateint can be loaded or written in web page (interface.html). 
Flask app ( Flasktest.py ) calls python code(test.py) that will analyse the input (the medical report) and return a probability of inclusion bassed on our pre-trained HAN (percentage).

## Contribution
Contributions and user feebdacks are very welcomed.

Contact: loic.verlingue@gustaveroussy.fr 
