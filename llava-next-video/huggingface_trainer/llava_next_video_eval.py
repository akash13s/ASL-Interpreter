import json
import os

import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Paths
OUTPUT_DIR = "./output/"
OUTPUT_FILE = "./output/generated_texts.json"
EVAL_FILE = "./output/evaluation_metrics.csv"


class EvaluationMetrics:
    def __init__(self, json_file_path, output_dir):
        """
        Initialize the evaluation metrics class.

        Args:
            json_file_path (str): Path to the JSON file containing true and generated text.
            output_dir (str): Directory to save the evaluation results.
        """
        self.json_file_path = json_file_path
        self.output_dir = output_dir
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        print(f"Initialized EvaluationMetrics with JSON file: {self.json_file_path}")
        self.data = self._load_and_fix_json()

    def _load_and_fix_json(self):
        """
        Load and parse the JSON file, accessing the `generated_texts` field.

        Returns:
            list: Parsed JSON data as a list of dictionaries.
        """
        print(f"Loading and checking JSON file: {self.json_file_path}")
        with open(self.json_file_path, 'r') as file:
            json_data = json.load(file)

        # Ensure the key `generated_texts` is present and contains data
        if "generated_texts" not in json_data or not isinstance(json_data["generated_texts"], list):
            raise ValueError("The JSON file does not contain a valid 'generated_texts' field.")

        data = json_data["generated_texts"]
        print(f"Loaded {len(data)} records from JSON file.")
        return data

    def compute_metrics(self, sample):
        """
        Compute ROUGE and BLEU metrics for a single sample.

        Args:
            sample (dict): A dictionary containing "true" and "generated" fields.

        Returns:
            dict: Metrics including ROUGE-1, ROUGE-2, ROUGE-L, and BLEU scores.
        """
        true_desc = sample["true"]
        gen_desc = sample["generated"]

        # ROUGE Scores
        rouge_scores = self.scorer.score(true_desc, gen_desc)

        # BLEU Score with smoothing (unigram and bigram)
        smoothing_function = SmoothingFunction().method1
        bleu_score = sentence_bleu(
            [true_desc.split()],
            gen_desc.split(),
            weights=(0.5, 0.5),
            smoothing_function=smoothing_function
        )

        print(f"Computed metrics for ID: {sample['id']} (Epoch: {sample['epoch']})")
        return {
            "epoch": sample["epoch"],
            "rouge1": rouge_scores['rouge1'].fmeasure,
            "rouge2": rouge_scores['rouge2'].fmeasure,
            "rougeL": rouge_scores['rougeL'].fmeasure,
            "bleu": bleu_score
        }

    def evaluate(self):
        """
        Perform evaluation by computing all metrics and saving results to a CSV file.

        Returns:
            str: Path to the saved CSV file.
        """
        print("Starting evaluation process...")

        # Compute metrics for each sample
        print("Computing metrics for each sample...")
        metric_results = [self.compute_metrics(sample) for sample in self.data]

        # Convert results to DataFrame
        df = pd.DataFrame(metric_results)
        print(f"Computed metrics for all {len(self.data)} samples.")

        # Calculate averages per epoch
        print("Calculating average metrics per epoch...")
        avg_per_epoch = df.groupby("epoch").mean(numeric_only=True).reset_index()

        # Save results with per-epoch averages
        avg_per_epoch.to_csv(EVAL_FILE, index=False)
        print(f"Per-epoch metrics calculated and saved to {EVAL_FILE}")


if __name__ == "__main__":
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize the evaluator
    evaluator = EvaluationMetrics(OUTPUT_FILE, OUTPUT_DIR)

    # Run the evaluation
    evaluator.evaluate()