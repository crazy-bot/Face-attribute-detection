## The Tiny Portraits Challenge Submission

### Goals of the challenge
Lets summarize the goal of this challenge from my perpective.
- Explore the Tiny portrait Dataset
    - data description, no of images, no of classes
    - check for class imblanace
    - check for missing values
    - check for outliers
    - visualize data
- Gender classification
    - train a model to classify gender 
    - evaluate the model by unseen test dataset within Tiny Portraits 
    - evaluate the model by unseen images from other sources
- Hair color classification
    - train a model to classify gender 
    - evaluate the model by unseen test dataset within Tiny Portraits 
    - evaluate the model by unseen images from other sources
- Combind the above two task by sinle model
    - multi task training
    - evaluate the model by unseen test dataset within Tiny Portraits 
    - evaluate the model by unseen images from other sources    
- Lastly, generate such tiny face images from scratch
    - generative model approach

### Explore the Tiny portrait Dataset
I have explored tha dataset in this notebook [data_exploration](solution/notebooks/data exploration.ipynb)
I have explained my observation and thoughts behind each step within the notebook.

### Dataset Preprocessing

1. Bad images filter: During dataset exploration, I found many non-face images, so I saved those file names in [bad_images](solution/asset/bad_images.txt)

2. I also found class imabalce. So, for gender and haircolor classification seperately, I have used stratified sampling technique to seperate train, validation and test set.

3. For multitask training I seperated random split.

4. For step 2 and 3, I have removed the bad images.

5. There were missing values ('n/a') in hair_color column. I found that dropping those columns improved the performance comapritively to classifying all of them as 'others'.So I finally dropped them for haircolor classification and multitask training.

In [datastat.py](solution/misc/datastat.py) the logic is implemented and all the seperated set has been saved in [asset](solution/asset) in the form of csv file.

### Environment setting
I have created a [requirements.txt](requirements.txt) to replicate the environment and install packages used for the task.

### Gender Classification
Potential amount of research works have been made so far in face cognition and face embedding learning. So, I decided to fine tune one of the state-of-the -art network for this task. 
I have finetuned all the layers of [facenet](https://github.com/timesler/facenet-pytorch) model which was trained on 'VGGFace2'. I found freezing initial layers giving very low performance because of different size and dataset mostly.

- cost function: CrossEntropy
- data sampler: Weighted sampling
- no of class: 2

#### Training
start training by below command:

´´´
    python -m mains.train_classifier \
    --dataroot '/data/suparna/workspace/TinyPortraits_thumbnails/' \
    --classification_type 'gender' \
    --exp_name gender_exp_2 \
    --batch_size 200 \
    --epochs 50 \
    --lr 0.001
´´´
see [utils.py](solution/utils/utils.py) for the description of each arguments.

#### Testing
start testing by below command:

´´´
CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=1 \
python -m mains.test_classifier \
    --dataroot '/data/suparna/workspace/TinyPortraits_thumbnails/' \
    --classification_type 'gender' \
    --exp_name gender_exp_2 \
    --batch_size 1 \
    --ckpt 'best' \
´´´
#### Prediction
start testing by below command:

´´´
CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=1 \
python -m mains.predict \
    --dataroot '/data/suparna/workspace/face_reasearch/smallfaces' \
    --classification_type 'gender' \
    --exp_name gender_exp_2 \
    --ckpt 'best' \
´´´
#### Result
1. all the model checkpoints will be saved in 'checkpoints/<exp_name>/models/
2. all the log files will be saved 'checkpoints/<exp_name>/*.log
3. tf-summarywriter are saved in runs
4. run.log contains training logs and evaluation metric for training and validation.
5. test.log contains testing evaluation on unseen Tiny Portraits set
6. pred.log contains prediction summary on unseen data from [sample_images](sample_images)
7. output of step 6 are also saved in 'checkpoints/<exp_name>/predictions_gender.csv'

Below is the classification report for test set:


**classification_report:** 

               precision    recall  f1-score   support

      female       0.99      0.98      0.98     16760
        male       0.97      0.98      0.97     10055

    accuracy                           0.98     26815
   macro avg       0.98      0.98      0.98     26815
weighted avg       0.98      0.98      0.98     26815
´´´

### Haircolor Classification 
I have finetuned all the layers of [facenet](https://github.com/timesler/facenet-pytorch) model which was trained on 'VGGFace2'. No of class is 5

- cost function: CrossEntropy
- data sampler: Weighted sampling
- no of class: 5

#### Training
start training by below command:

´´´
CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=1 \
python -m mains.train_classifier \
    --dataroot '/data/suparna/workspace/TinyPortraits_thumbnails/' \
    --classification_type 'haircolor' \
    --exp_name hair_exp_3 \
    --batch_size 200 \
    --epochs 50 \
    --lr 0.001
´´´
see [utils.py](solution/utils/utils.py) for the description of each arguments.

#### Testing
start testing by below command:

´´´
CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=1 \
python -m mains.test_classifier \
    --dataroot '/data/suparna/workspace/TinyPortraits_thumbnails/' \
    --classification_type 'haircolor' \
    --exp_name hair_exp_3 \
    --batch_size 200 \
    --ckpt 'best' \
´´´
#### prediction
start prediction by below command:

´´´
CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=1 \
python -m mains.predict \
    --dataroot '/data/suparna/workspace/face_reasearch/smallfaces' \
    --classification_type 'haircolor' \
    --exp_name hair_exp_3 \
    --ckpt 'best' \
´´´
#### Result
1. all the model checkpoints will be saved in 'checkpoints/<exp_name>/models/
2. all the log files will be saved 'checkpoints/<exp_name>/*.log
3. tf-summarywriter are saved in runs
4. run.log contains training logs and evaluation metric for training and validation.
5. test.log contains testing evaluation on unseen Tiny Portraits set
6. pred.log contains prediction summary on unseen data from [sample_images](sample_images)
7. output of step 6 are also saved in 'checkpoints/<exp_name>/predictions_haircolor.csv'

Below is the classification report for test set:

**classification_report:**

               precision    recall  f1-score   support

        bald       0.79      0.80      0.80       363
       black       0.93      0.93      0.93      6490
       blond       0.90      0.92      0.91      4090
       brown       0.88      0.87      0.87      5884
        gray       0.87      0.81      0.84       933

    accuracy                           0.90     17760
   macro avg       0.87      0.87      0.87     17760
weighted avg       0.90      0.90      0.90     17760
´´´
### Multi task training
For training a single model for multiple tasks, I have trained multi head network with common feature extraction layers. Because here the two tasks are two similar classification that needs common understanding of the image feature, we can do this.

- cost function: Weighted CrossEntropy
- data sampler: Random sampling
- no of class for gender: 2
- no of class for haircolor: 5

#### Training
start training by below command:

´´´
CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=1 \
python -m mains.train_multitask \
    --dataroot '/data/suparna/workspace/TinyPortraits_thumbnails/' \
    --classification_type 'multitask' \
    --exp_name multitask_exp_2 \
    --batch_size 200 \
    --epochs 50 \
    --lr 0.001
´´´
see [utils.py](solution/utils/utils.py) for the description of each arguments.

#### Testing
start testing by below command:

´´´
CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=1 \
python -m mains.test_multitask \
    --dataroot '/data/suparna/workspace/TinyPortraits_thumbnails/' \
    --classification_type 'multitask' \
    --exp_name multitask_exp_2 \
    --batch_size 200 \
    --ckpt 'best' \

´´´
#### prediction
start prediction by below command:

´´´
CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=1 \
python -m mains.predict \
    --dataroot '/data/suparna/workspace/face_reasearch/smallfaces' \
    --classification_type 'multitask' \
    --exp_name multitask_exp_2 \
    --ckpt 'best'
´´´
#### Result
1. all the model checkpoints will be saved in 'checkpoints/<exp_name>/models/
2. all the log files will be saved 'checkpoints/<exp_name>/*.log
3. tf-summarywriter are saved in runs
4. run.log contains training logs and evaluation metric for training and validation.
5. test.log contains testing evaluation on unseen Tiny Portraits set
6. pred.log contains prediction summary on unseen data from [smallfaces](smallfaces)
7. output of step 6 are also saved in 'checkpoints/<exp_name>/predictions_multitask.csv'

Below is the classification report for test set:

´´´
**classification_report of gender classification:**

               precision    recall  f1-score   support

      female       0.99      0.99      0.99     11561
        male       0.97      0.97      0.97      6199

    accuracy                           0.98     17760
   macro avg       0.98      0.98      0.98     17760
weighted avg       0.98      0.98      0.98     17760

**classification_report of hair color classification:**

               precision    recall  f1-score   support

        bald       0.82      0.53      0.64       350
       black       0.93      0.92      0.92      6466
       blond       0.93      0.86      0.89      4128
       brown       0.83      0.91      0.87      5865
        gray       0.82      0.80      0.81       951

    accuracy                           0.89     17760
   macro avg       0.87      0.80      0.83     17760
weighted avg       0.89      0.89      0.89     17760

´´´
### Sample Images Evaluation
I have gathered 10 different images from different sources including different ethnicity than training data and black and white colored image.

3 female are misclassified as male. They are in different skin color, age groups, 

hair color is sometimes confusing between brown and gray

### Future scope (not done here)
- hyperparameter search
- experimentation with different augmentation
- collecting more diverse data for low precision classes

### Image Generation from scratch
This problem can be inherently solved by adversarial network. So I have trained one of the state-of-art network StyleGan2 that I had previously worked with.

It had become popular because of its abílity in generating high resolution images and controlling the style of generated images. 

Our task is much simpler. I have padded the images into 128*128 and trained the styleGAN2. As preprocess step I retrieved the original size.

The training code given here which is taken from official repository [stylegan2-ada](https://github.com/NVlabs/stylegan2-ada-pytorch).

The intermmediate outputs are saved in [this folder](stylegan2-ada-pytorch/training-runs/00006-TinyPortraits_thumbnails-auto1-resumecustom)

[generated output](stylegan2-ada-pytorch/out)

For training and generating image refer [StyleGAN2 README](stylegan2-ada-pytorch/README.md)

To generate with different seed:
´´´
cd stylegan2-ada-pytorch

python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \
    --network=training-runs/00006-TinyPortraits_thumbnails-auto1-resumecustom/network-snapshot-003000.pkl

´´´

The network was trained for 15 hours.










