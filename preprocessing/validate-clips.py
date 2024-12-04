import pandas as pd
import os
from moviepy.editor import VideoFileClip
from tqdm import tqdm

# Constants for file paths
CSV_FILE = '/scratch/rr4577/translation/train.csv'
VIDEO_DIR = '/scratch/as18464/raw_videos/'

ALLOWED_DURATION = 30

# Function to check if the file exists and validate duration
def validate_clip(row, video_dir):
    # Construct the file path
    file_path = os.path.join(video_dir, f"{row['SENTENCE_NAME']}.mp4")

    # Check if the file exists
    if not os.path.exists(file_path):
        return False

    # Check the duration of the .mp4 file
    try:
        with VideoFileClip(file_path) as video:
            video_duration = video.duration 
            if row['duration'] <= ALLOWED_DURATION and video_duration <= ALLOWED_DURATION:
                return True
            else:
                return False
    except Exception as e:
        print(f"Error with file {file_path}: {e}")
        return False

# Main function to process the DataFrame
def main():
    # Load the CSV file with a custom delimiter
    df = pd.read_csv(CSV_FILE, delimiter='\t')

    # Calculate the duration for each clip
    df['duration'] = df['END_REALIGNED'] - df['START_REALIGNED']

    # List to store valid rows
    valid_rows = []

    # Iterate over the DataFrame with a progress bar using tqdm
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Validating Clips"):
        print(f"processing row: {index}")
        if validate_clip(row, VIDEO_DIR):
            valid_rows.append(row)
            print(f"valid row: {row['SENTENCE_NAME']}")
        else:
            print(f"invalid row: {row['SENTENCE_NAME']}")
            
    # Create a new DataFrame with only valid rows
    valid_df = pd.DataFrame(valid_rows)

    # Save the new DataFrame to a new CSV file
    valid_df.to_csv('valid_clips.csv', index=False)

    print("Filtered DataFrame saved to 'valid_clips.csv'")

# Entry point for the script
if __name__ == "__main__":
    main()
