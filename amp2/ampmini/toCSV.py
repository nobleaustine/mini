import os
import pandas as pd
from Bio import SeqIO

# List of labels based on subfolder names
labels = ['antibacterial','antifungal','antiviral','anticancer','antigram-positive','antigram-negative',
          'anti_mammalian_cells','antihiv','antimrsa','antiparasitic','antibiofilm',
          'chemotactic','endotoxin','insecticidal',]
        #   'antitb','anurandefense','cytotoxic','hemolytic',
        #   'antimalarial','anticandida','antiplasmodial','antiprotozoal','toxic',"antioxidant"]

# Function to extract sequences from a FASTA file
def get_sequences_from_fasta(fasta_file, used_ids):
    sequences = []
    with open(fasta_file, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            seq_id = str(record.id)
            original_id = seq_id
            suffix = 1
            while seq_id in used_ids:
                seq_id = f"{original_id}X{suffix}"
                suffix += 1
            used_ids.add(seq_id)
            sequences.append((seq_id, str(record.seq)))
    return sequences

# Function to process all subfolders and create the CSV
def process_fasta_files(root_dir, output_csv):
    data = {}  # Dictionary to store sequences and their labels
    used_ids = set()  # Track used sequence IDs

    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)
        
        # Skip if not a directory or not in labels list
        if not os.path.isdir(subfolder_path) or subfolder not in labels:
            continue
        
        pos_file = os.path.join(subfolder_path, "pos.fasta") #_cdhit_100
        neg_file = os.path.join(subfolder_path, "neg.fasta") #_cdhit_100clear
        

        # Process POSITIVE FASTA
        if os.path.exists(pos_file):
            
            for seq_id, sequence in get_sequences_from_fasta(pos_file, used_ids):
                if sequence not in data:
                    data[sequence] = {'id': seq_id, 'sequence': sequence, **{label: 0 for label in labels}}
                data[sequence][subfolder] = 1
        

        # Process NEGATIVE FASTA
        if os.path.exists(neg_file):
            for seq_id, sequence in get_sequences_from_fasta(neg_file, used_ids):
                if sequence not in data:
                    data[sequence] = {'id': seq_id, 'sequence': sequence, **{label: 0 for label in labels}}

        print(f"Processed {subfolder}: {len(data)} sequences")

    # Convert to DataFrame and save
    df = pd.DataFrame(data.values())
    df.to_csv(output_csv, index=False)
    print(f"CSV file saved: {output_csv}")

# Run the script
root_directory = "/cluster/home/austinen/mini/ampmini/data/AMP_2nd_test"
output_csv_file = "/cluster/home/austinen/mini/ampmini/data/new_AMP_sequences_test.csv"

process_fasta_files(root_directory, output_csv_file)
