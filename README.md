# Sign-Language-Translation

One Paragraph of project description goes here

## Usage

1. ```pyenv virtualenv 3.6.5 my-virtual-env-3.6.5``` ; create a virtual environment
2. ```pyenv activate my-virtual-env-3.6.5 ``` ; activate the virtual environment
3. ```pip install -r requirements.txt``` ; Install dependencies

## Prepare data 

```python sign_langauge_detection.py prepare_data ```

Prepares data for training the model. It would output
the following files in the **/data** folder. 

1. test.h5 
2. test_id.json 
3. test_labels.h5  
4. train.h5 
5. train_id.json 
6. train_labels.h5 
7. validation.h5 
8. valid_id.json 
9. valid_labels.h5 

Distribution of data: 

1. Training set : 80% 
2. Validation set : 10% 
3. Test set : 10%

## Training your own model 
```python sign_langauge_detection.py train ```

Make sure all the relevant files are in **/data** folder. 

NOTE: Training our own InceptionV3 can be computationally expensive on GPU. I used Google Colab GPU to train
the model on 100 epochs. 

After training, it would output model checkpoint file in **/checkpoint_model** folder. 

It would also export model in 
**export_model/inception_v3/1** folder.  

Tensorboard logs would output at **/logs** folder which can be viewed 
using the following command:
 
```tensorboard --logdir="./logs/inception_logs"```

## Start demo 

The demo can be started using the following command:

```python sign_langauge_detection.py``` 

## Credits 

1. [Inception model](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)

