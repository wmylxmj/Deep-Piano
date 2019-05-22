# Deep-Piano
#### Play The Piano With Deep Learning

[view demo here](https://github.com/wmylxmj/Deep-Piano/edit/master/outputs/)

### prepare your dataset
- collect some midi files and put them in the midi folder
- run prepare.py, we will count the frequency at which notes are pressed

<div align="center">
  <img src="images/sequence.png" height="320" width="400" >
  <img src="images/count.png" height="320" width="400" >
</div>

### creat the model
- the model uses deep one-dimensional residual dense convolution network to extract time series characteristics

<div align="center">
  <img src="images/RDB.png" height="400" width="600" >
</div>

- the model predicts the state of 88 keys in 1/64 time at a time
- the model generation process is shown in the following figure

<div align="center">
  <img src="images/Conv1D.png" height="400" width="600" >
</div>

### train the model
- run train.py

### generate piano melody
- run predict.py
