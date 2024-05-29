#!/bin/bash
#
#SBATCH -D/gpfs/projects/HeinzGroup/subregular-learning/
#SBATCH --job-name=train_langs
#SBATCH --output=train_langs.out
#SBATCH --partition=short-40core
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=sam.vanderpoel@stonybrook.edu

python src/lang_names.py --action train_done --model_dir models > train_done.txt
python src/lang_names.py --action data_gen_done --avoid train_done.txt > train_langs.txt
python src/gen_trials.py

