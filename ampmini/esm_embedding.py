import torch
import pandas as pd
import os
import numpy as np
from transformers import AutoTokenizer, EsmForProteinFolding
from tqdm import tqdm
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"    

# Parameters
csv_path = "/cluster/home/austinen/mini/ampmini/data/new_AMP_sequences_test.csv"         # CSV with columns: "id", "sequence"
sequence_column = "sequence"
id_column = "id"
output_dir = "./data/esmfold_test_features/"
batch_size = 16

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load model and tokenizer
model = EsmForProteinFolding.from_pretrained("../esmfold_v1_local", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("../esmfold_v1_local", local_files_only=True)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Read CSV
df = pd.read_csv(csv_path)
sequences = df[sequence_column].tolist()
ids = df[id_column].tolist()

# Process in batches
for i in tqdm(range(0, len(sequences), batch_size)):
    batch_seqs = sequences[i:i+batch_size]
    batch_ids = ids[i:i+batch_size]

    inputs = tokenizer(
            batch_seqs, 
            return_tensors="pt", 
            add_special_tokens=False, 
            padding="max_length", 
            truncation=True,
            max_length=100)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    s_s = outputs.s_s.cpu().numpy()        # (B, L, D)
    s_z = outputs.s_z.cpu().numpy()        # (B, L, L, D)
    states = outputs.states.cpu().numpy()  # (B, L, D)
    states = states.transpose(1, 0, *range(2, states.ndim))
    print("batch+_id",len(batch_ids))
    print("s_s",s_s.shape)
    print("s_z",s_z.shape)
    print("states",states.shape)
    # Save outputs
    for j, sample_id in enumerate(batch_ids):
        filename = f"{sample_id}.npz"
        filepath = os.path.join(output_dir, filename)

        np.savez_compressed(filepath, s_s=s_s[j], s_z=s_z[j], states=states[j])
