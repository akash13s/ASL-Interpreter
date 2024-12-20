# ASL-Interpreter

We want to build a model to interpret continuous American Sign Language (ASL) signing into English text. For this use case, we conducted experiments for fine-tuning large language vision models, specifically:

1. **LLaVA-NeXT-Video**
2. **Video-LLaVA**

## Dataset

We are using the `How2Sign` dataset which is ASL video footage aligned with English sentences. It includes RGB videos, green-screen frontal and side views, and 3D keypoints (hand, body, face). We focused on RGB frontal-view video data for fine-tuning to manage computational constraints.

[How2Sign Dataset](https://how2sign.github.io/)

## Project Structure

- `data`: Contains the cleaned CSV file used as the source dataset.
- `data_profiling`: Includes the code for data cleaning.
- `llava-next-video`: Scripts for fine-tuning the LLaVA-NeXT-Video model on the How2Sign dataset, along with quantitative analysis of the trained model.
- `video-llava`: Scripts for fine-tuning the Video-LLaVA model on the How2Sign dataset, as well as inference scripts for the trained model.

---

## Installing Dependencies

```bash
pip install -r requirements.txt
```


## Running LLaVA-NeXT-Video

### 1. Perform Fine-Tuning

Navigate to the `huggingface_trainer` directory within `llava-next-video` and execute the following command:

```bash
cd llava-next-video/huggingface_trainer
sbatch train.sh
```

We used slurm jobs to trigger the training jobs.

#### Outputs:
- A `logs` folder will be created to store training logs.
- An `output` directory will be generated to store checkpoints from training.
- A `generated_texts.csv` file will be created for validation purposes.

#### `generated_texts.csv` Columns:
1. **id**: Incremental ID for each data item.
2. **video_id**: Unique identifier for the video clip, also present in `valid_clips.csv`.
3. **generated**: The text generated by the model for the specific clip.
4. **true**: The expected text for the specific clip.
5. **epoch**: The epoch at which the evaluation occurred.

### 2. Generate Validation Scores

Run the evaluation script to calculate validation scores:

```bash
cd llava-next-video/huggingface_trainer
python llava_next_video_eval.py
```

#### Outputs:
- A `validation_scores.csv` file will be generated containing the following metrics after every epoch:
  - **ROUGE-1**
  - **ROUGE-2**
  - **ROUGE-L**
  - **BLEU**

---

## Running Video-LLaVA

### 1. Perform Fine-Tuning

Navigate to the `video-llava` directory and execute the following command:

```bash
cd video-llava
sbatch train.sh
```

### 2. Perform Inference

Inference for the Video-LLaVA model can be performed using the Jupyter notebook located at:

```text
video-llava/inference.ipynb
```

---
