#!/bin/bash
#----------------------------------
# Specifying grid engine options
#----------------------------------
#$ -S /bin/bash  
# the working directory where the commands below will
# be executed: (make sure to specify)
#$ -wd /data/users/sstauden/dev/SQuAD
#
# logging files will go here: (make sure to specify)
#$ -e /data/users/sstauden/dev/log/ -o /data/users/sstauden/dev/log/
#  
# Receive a mail when your job is done: (currently not working)
#$ -m e  
#$ -M sstauden@lsv.uni-saarland.de  

# activate environment
export PATH="/nethome/sstauden/miniconda3/bin:$PATH"
source activate master_env
#----------------------------------
#  Running some bash commands 
#----------------------------------
pwd
echo "Generating InferSent Embeddings"
#----------------------------------
# Running your code

python create_emb.py

echo "Finished"
