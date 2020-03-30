#!/bin/bash
#SBATCH -N 1  # number of nodes
#SBATCH -n 8
#SBATCH -t 6-00:00:00   # time in d-hh:mm:ss
#SBATCH -p serial       # partition 
#SBATCH -q normal       # QOS
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user=zwinzurk@asu.edu # Mail-to address

# Kgalagadi

wget https://lilablobssc.blob.core.windows.net/snapshot-safari/KGA/KGA_S1.lila.zip -P ../../../scratch/zwinzurk/wild/datasets/KGA_S1.zip

# Camdeboo

wget https://lilablobssc.blob.core.windows.net/snapshot-safari/CDB/CDB_S1.lila.zip -P ../../../scratch/zwinzurk/wild/datasets/CDB_S1.zip

# Kruger

wget https://lilablobssc.blob.core.windows.net/snapshot-safari/KRU/KRU_S1.lila.zip -P ../../../scratch/zwinzurk/wild/datasets/KRU_S1.zip

# Mountain Zebra

wget https://lilablobssc.blob.core.windows.net/snapshot-safari/MTZ/MTZ_S1.lila.zip -P ../../../scratch/zwinzurk/wild/datasets/MTZ_S1.zip

# Serengeti S02

wget https://lilablobssc.blob.core.windows.net/snapshotserengeti-v-2-0/SnapshotSerengeti_S02_v2_0.zip -P ../../../scratch/zwinzurk/wild/datasets/S02.zip

# Serengeti S04

wget https://lilablobssc.blob.core.windows.net/snapshotserengeti-v-2-0/SnapshotSerengeti_S04_v2_0.zip -P ../../../scratch/zwinzurk/wild/datasets/S04.zip

# Serengeti S05

wget https://lilablobssc.blob.core.windows.net/snapshotserengeti-v-2-0/SnapshotSerengeti_S05_v2_0.zip -P ../../../scratch/zwinzurk/wild/datasets/S05.zip

# Serengeti S06

wget https://lilablobssc.blob.core.windows.net/snapshotserengeti-v-2-0/SnapshotSerengeti_S06_v2_0.zip -P ../../../scratch/zwinzurk/wild/datasets/S06.zip

