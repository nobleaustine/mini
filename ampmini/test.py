from Bio import SeqIO

# path1="./data/AMP_1stage/AMPs_train_cdhit_40.fasta"
# path1 = "./models/data/AMP_1stage/AMPs_train_cdhit_40.fasta"
# fas_id=[]
# fas_seq=[]
# labels=[]

# for seq_record in SeqIO.parse(path1, "fasta"):
#     fas_seq.append(str(seq_record.seq).upper())
#     fas_id.append(str(seq_record.id))
#     labels.append(1)
# print(len(fas_seq))

from ampmini.data_feature_n import onehot_embedding
