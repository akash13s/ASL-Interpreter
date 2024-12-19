import os
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Paths
OUTPUT_FILE = "./output/generated_texts.csv"
EVAL_FILE = "./output/validation_scores.csv"


class EvaluationMetrics:
    def __init__(self, csv_file_path):
        """
        Initialize the evaluation metrics class.

        Args:
            csv_file_path (str): Path to the CSV file containing true and generated text.
        """
        self.csv_file_path = csv_file_path
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        print(f"Initialized EvaluationMetrics with CSV file: {self.csv_file_path}")
        self.data = self._load_and_fix_csv()

    def _load_and_fix_csv(self):
        """
        Load and parse the CSV file.

        Returns:
            pd.DataFrame: Loaded data as a DataFrame.
        """
        print(f"Loading and checking CSV file: {self.csv_file_path}")
        data = pd.read_csv(self.csv_file_path)

        # Ensure the CSV has the required columns
        required_columns = ["epoch", "id", "video_id", "generated", "true"]
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"The CSV file must contain the following columns: {required_columns}")

        print(f"Loaded {len(data)} records from CSV file.")
        return data

    def compute_metrics(self, sample):
        """
        Compute ROUGE and BLEU metrics for a single sample.

        Args:
            sample (pd.Series): A row from the DataFrame containing "true" and "generated" fields.

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
        metric_results = [self.compute_metrics(row) for _, row in self.data.iterrows()]

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
    # Initialize the evaluator
    evaluator = EvaluationMetrics(OUTPUT_FILE)

    # Run the evaluation
    evaluator.evaluate()
