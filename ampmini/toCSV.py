import os
import pandas as pd
from Bio import SeqIO

# List of labels based on subfolder names
labels = ['antibacterial','antigram-positive','antigram-negative','antifungal','antiviral',
          'anti_mammalian_cells','antihiv','antibiofilm','anticancer','antimrsa','antiparasitic',
          'hemolytic','chemotactic','antitb','anurandefense','cytotoxic',
          'endotoxin','insecticidal','antimalarial','anticandida','antiplasmodial','antiprotozoal','toxic',"antioxidant"]

# Function to extract sequences from a FASTA file
def get_sequences_from_fasta(fasta_file):
    sequences = []
    with open(fasta_file, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            sequences.append((record.id, str(record.seq)))  # (id, sequence)
    return sequences

# Function to process all subfolders and create the CSV
def process_fasta_files(root_dir, output_csv):
    data = {}  # Dictionary to store sequences and their labels

    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)
        
        # Skip if not a directory or not in labels list
        if not os.path.isdir(subfolder_path) or subfolder not in labels:
            continue
        
        pos_file = os.path.join(subfolder_path, "pos_cdhit_100.fasta")
        neg_file = os.path.join(subfolder_path, "neg_cdhit_100.fasta")

        # Process POSITIVE FASTA: Add 1 for the current subfolder
        if os.path.exists(pos_file):
            for seq_id, sequence in get_sequences_from_fasta(pos_file):
                if sequence not in data:
                    # Initialize with all labels as 0
                    data[sequence] = {'id': seq_id, 'sequence': sequence, **{label: 0 for label in labels}}
                data[sequence][subfolder] = 1  # Mark presence in positive file

        # Process NEGATIVE FASTA: Only ensure sequence exists, but do NOT mark 1
        if os.path.exists(neg_file):
            for seq_id, sequence in get_sequences_from_fasta(neg_file):
                if sequence not in data:
                    # Initialize with all labels as 0, but don't change labels since it's from neg file
                    data[sequence] = {'id': seq_id, 'sequence': sequence, **{label: 0 for label in labels}}

    # Convert dictionary to DataFrame and save as CSV
    df = pd.DataFrame(data.values())
    df.to_csv(output_csv, index=False)
    print(f"CSV file saved: {output_csv}")

# Run the script
root_directory = "/cluster/home/austinen/mini/ampmini/data/AMP_2nd_train"  # Change to your actual path
output_csv_file = "/cluster/home/austinen/mini/ampmini/AMP_sequences.csv"

process_fasta_files(root_directory, output_csv_file)
