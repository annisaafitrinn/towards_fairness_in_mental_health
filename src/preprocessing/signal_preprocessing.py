from utils import *  # if function is in utils.py

def main():
    eeg_path = "data/eeg_128_channels"
    segmented_path = "data/segmented_eeg/"
    merged_file_path = "data/segmented_eeg/merged_eeg_data.npy"
    labels_file_path = "data/segmented_eeg/labels.npy"

    process_and_segment(eeg_path, segmented_path)
    merge_segmented_data(segmented_path, merged_file_path, labels_file_path)

if __name__ == "__main__":
    main()