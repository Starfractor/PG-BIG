# PG-BIG: Personalize Guidance for Biomechanically Informed GenAI

## Summary
PG-BIG is a framework for personalized guidance in biomechanically informed generative AI, focusing on motion modeling and evaluation using VQ-VAE and surrogate models.

## Table of Contents
- [Summary](#summary)
- [Installation](#installation)
- [VQ-VAE Training](#vq-vae-training)
- [Profile Prior Training](#profile-prior-training)
- [Surrogate Training](#surrogate-training)
- [Guidance](#guidance)
- [Evaluation Metrics](#evaluation-metrics)

## Dependencies
1. Clone the repository:
    ```
    git clone https://github.com/your-org/PG-BIG.git
    cd PG-BIG
    ```
2. Install dependencies using Conda or pip:
    ```
    pip install -r requirements.txt
    ```
    ```
    conda create env -f enviornment.yaml
    ```
## Datasets
You can download/create the datasets following the instructions below.
### Three-Dimensional Motion Capture Data of a Movement Screen from 183 Athletes
1. Navigate to `dataset` and create a new directory called `three_dimensional_motion_capture`:
    ```
    cd dataset
    mkdir -p three_dimensional_motion_capture
    ```
2. Go to the [Figshare collection page](https://springernature.figshare.com/collections/Three-Dimensional_Motion_Capture_Data_of_a_Movement_Screen_from_183_Athletes/6014509).

   Download both:
   - [Three-Dimensional Motion Capture Data of All Athletes](https://springernature.figshare.com/articles/dataset/Three-Dimensional_Motion_Capture_Data_of_All_Athletes/19879894?backTo=%2Fcollections%2FThree-Dimensional_Motion_Capture_Data_of_a_Movement_Screen_from_183_Athletes%2F6014509&file=39075392)
   - [Participants' Information and Sampling Frequency](https://springernature.figshare.com/articles/dataset/Participants_information_and_sampling_frequency/19879891?backTo=/collections/Three-Dimensional_Motion_Capture_Data_of_a_Movement_Screen_from_183_Athletes/6014509)

   Place both zip files inside the `three_dimensional_motion_capture` directory. Then unzip both files:
    ```
    unzip Kinematic_Data.zip -d Kinematic_Data
    unzip Participants\ Info.zip -d Participants\ Info
    ```
3. Fit the dataset markers to a Rajapoal Skeleton:

   - Download the [Rajapoal OpenSim Model](https://simtk.org/frs/?group_id=773) by opening the link in your browser.
     ```
     "$BROWSER" https://simtk.org/frs/?group_id=773
     ```
   - Place the downloaded zip file (`FullBodyModel-4.0.zip`) into the `dataset` directory.
   - Unzip the model:
     ```
     unzip FullBodyModel-4.0.zip -d FullBodyModel-4.0
     ```
   - Run the retargeting algorithm to save skeletal models for each subject inside `dataset/183_athletes`:
     ```
     python retarget_dataset.py
     ```
   - To speed up the process, you can use multiple workers:
     ```
     python retarget_dataset.py --num_workers <num_workers>
     ```
     Replace `<num_workers>` with the number of workers you want to use.

## VQ-VAE Training
Train the VQ-VAE model for motion representation:
```
python3 train_vq.py --batch-size 256 --lr 2e-4 --total-iter 300000 --lr-scheduler 200000 --nb-code 512 --down-t 2 --depth 3 --dilation-growth-rate 3 --out-dir output --dataname mcs --vq-act relu --quantizer ema_reset --loss-vel 0.5 --recons-loss l1_smooth --exp-name 183_athletes
```
If you have 2+ CUDA GPUs, you can utilize DeepSpeed for faster training. Fill in `num_gpus` based on the number of GPUs you'll be using for training:
```
deepspeed --num_gpus=<num_gpus> train_vqvae.py --batch-size 256 --window-size 512 --lr 2e-4 --total-iter 300000 --lr-scheduler 200000 --nb-code 512 --down-t 2 --depth 3 --dilation-growth-rate 3 --out-dir output --dataname 183_athletes --vq-act relu --quantizer ema_reset --loss-vel 0.5 --recons-loss l1_smooth --exp-name VQVAE
```
Supported datasets: `183_athletes`, `addbiomechanics`.

## Profile Prior Training
Train the profile prior model for personalized guidance:
```
python train_profile_prior.py 
```

## Surrogate Training
Train surrogate models for biomechanical evaluation:
```
python train_surrogate.py --dataname <dataset>
```

## Guidance
Use the guidance module to generate or refine motions based on personalized profiles:
```
python guidance.py --input <motion_file> --profile <profile_file>
```

## Evaluation Metrics
Evaluate generated motions using built-in metrics:
- Reconstruction Loss
- Perplexity
- Commitment Loss
- Temporal Consistency

Run evaluation:
```
python evaluate.py --model <model_checkpoint> --dataset <dataset>
```