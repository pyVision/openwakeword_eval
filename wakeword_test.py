from openwakeword.model import Model
from glob import iglob
import os
from openwakeword.utils import bulk_predict
import openwakeword
import sys

def main(samples_path):
    """
    Main function to process WAV files for wakeword detection and calculate false accepts.

    Args:
    samples_path (str): The path where the sample WAV files are located.
    """

    # Create a glob pattern to recursively find all files within the specified directory
    rootdir_glob = samples_path + "/**/*" # Note the added asterisks for recursive search

    # Get a list of absolute file paths for all files found using the glob pattern
    file_path_list = [f for f in iglob(rootdir_glob, recursive=True) if os.path.isfile(f)]

    # Download the wakeword models, specifically the "hey_mycroft_v0.1" model
    openwakeword.utils.download_models(model_names=["hey_mycroft_v0.1"])

    # Get predictions for each audio file using multiprocessing
    # `bulk_predict` returns the prediction score for each frame of the audio
    prediction = bulk_predict(
        file_paths = file_path_list,
        wakeword_models = ["models/hey_miko_200_checkpoint_Big_DNN_2s_July17.onnx"],
        inference_framework="onnx",
        ncpu = 14 # Number of CPU cores to use for parallel processing
    )

    total_count = 0 # Initialize total count of processed files
    false_accepts = [] # List to keep track of files with false accepts

    # Iterate through the prediction results
    for key, value in prediction.items():
        count = 0
        total_count += 1 # Increment the total count
        # Iterate through the prediction values for each file
        for v in value:
            #print(v)
            # Check if the prediction score for "hey_mycroft_v0.1" exceeds the threshold (0.8)
            if v["hey_miko_model16"] > 0.8:
                count += 1
        # If any false accepts are found for a file, add it to the list
        if count > 0:
            false_accepts.append({key: count})

    # Print the percentage of files with false accepts
    print("False accepts are", len(false_accepts) * 100 / total_count, "%")
    
    return false_accepts,total_count

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <samples_path>")
        sys.exit(1)
    
    samples_path = sys.argv[1]

    rootdir_glob = samples_path+"/*"  # Note the added asterisks for recursive search

    # Get a list of absolute file paths for all files found using the glob pattern
    file_path_list = [f for f in iglob(rootdir_glob, recursive=False) if os.path.isdir(f)]

    print("file_path_list",file_path_list)
    d=[]
    for k in file_path_list:
        print("processing",k)
        fa,total_count=main(k)
        print("results",len(fa),total_count)
        d.append({"dataset":k,"FA":len(fa),"total":total_count})

    print("results are ",d)