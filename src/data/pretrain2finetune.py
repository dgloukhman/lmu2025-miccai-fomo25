import multiprocessing as mp
import os
import re
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

pattern = re.compile(r"sub_(\d+)_ses_(\d+)_(.+?)(?:_\d+)?\.npy")


#Throwaway script to convert pre-train to finetuning data

def convert_to_nifti(numpy_array, output_filename):
    """
    Converts a 3D numpy array to a .nii.gz file.

    Args:
        numpy_array (np.ndarray): The 3D numpy array to convert.
        output_filename (str): The path to save the .nii.gz file.
    """
    affine = np.eye(4)
    nifti_file = nib.Nifti1Image(numpy_array, affine)
    nib.save(nifti_file, output_filename)


def process_and_stack_subjects(input_directory, output_directory):
    """
    Finds subjects with T2 FLAIR, DWI, and T2/SWI scans for each session,
    stacks their numpy arrays, and saves them to a new directory.
    """
    subjects_scans = defaultdict(list)
    subjects_types = defaultdict(set)
    # Updated pattern to capture session ID
    pattern = re.compile(r"sub_(\d+)_ses_(\d+)_(.+?)(?:_\d+)?\.npy")

    try:
        files = os.listdir(input_directory)
    except FileNotFoundError:
        print(f"Error: Directory not found at {input_directory}")
        return

    for filename in files:
        match = pattern.match(filename)
        if match:
            subject_id = match.group(1)
            session_id = match.group(2)
            scan_type_key = match.group(3)
            scan_path = os.path.join(input_directory, filename)
            scan_type = None
            if "flair" in scan_type_key:
                scan_type = "FLAIR"
            elif "dwi" in scan_type_key:
                scan_type = "DWI"
            elif "t2" in scan_type_key:
                scan_type = "T2"
            elif "swi" in scan_type_key:
                scan_type = "SWI"

            if scan_type:
                # Use a tuple of (subject, session) as the key
                subject_session_id = (subject_id, session_id)
                subjects_types[subject_session_id].add(scan_type)
                subjects_scans[subject_session_id].append(scan_path)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    stacked_count = 0
    # Iterate over each subject-session pair
    subjects_types = {k: v for k, v in subjects_types.items() if len(v) > 2}

    item_list = [(subjects_scans[k], v) for k, v in subjects_types.items()]

    with mp.Pool(mp.cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap(process_sub, item_list),
                total=len(item_list),
                desc="desc",
            )
        )

    print(
        f"\nFinished processing. Stacked scans for {stacked_count} subject-sessions."
    )


def process_sub(item):
    output_directory = "pre-custom"
    flist, scan_types = item
    if (
        "FLAIR" in scan_types
        and "DWI" in scan_types
        and (("T2" in scan_types) or ("SWI" in scan_types))
    ):
        # print(f"Processing {len(flist)} files for {scan_types}")
        for fp in flist:
            fname = fp.split("/")[-1]
            if match := pattern.match(fname):
                subject_id = match.group(1)
                session_id = match.group(2)
                scan_type_key = match.group(3)

                dir_path = Path(
                    output_directory + "/sub_" + subject_id + "/ses_1"
                )
                dir_path.mkdir(parents=True, exist_ok=True)
                dir_path = str(dir_path) + f"/{scan_type_key}.nii.gz"

                arr = np.load(fp, allow_pickle=True)
                convert_to_nifti(arr, dir_path)
            else:
                print(fname)
    return True


if __name__ == "__main__":
    data_directory = "/dss/mcmlscratch/04/ra58seq2/preprocessed/FOMO60k"
    output_dir = "/dss/mcmlscratch/04/ra58seq2/pre-custom"
    process_and_stack_subjects(data_directory, output_dir)
