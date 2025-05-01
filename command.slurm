#!/bin/bash
#SBATCH -J 3DU-train-semi
#SBATCH -p a100
#SBATCH -o 3DUtrain_1024.out
#SBATCH -e 3DUtrain_1024.err
#SBATCH -N 1 
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=end
#SBATCH --mail-user=zhyfself@163.com

module load miniconda3 cuda/11.6
source activate yf3DU_39_2

nvidia-smi

export PYTHONPATH="/dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG:$PYTHONPATH"

python /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/Main.py --m preprocess --pre /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/experiment/MSD_liver/data

python /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/Main.py --m train --d /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/experiment/MSD_liver/data/fold_0.json --o /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/experiment/MSD_liver/result/se_U_fold_0 --pd /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/experiment/MSD_liver/data/preprocess_data --f 0 --p "(64,192,192)"

python /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/Main.py --m predict --d /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/experiment/MSD_liver/data/predict_semi.json --o /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/experiment/MSD_liver/result/se_U_fold_0/predict --f 0 --c /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/experiment/MSD_liver/result/se_U_fold_0/checkpoint_best.pth --p "(64,192,192)"

python /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/Main.py --m evaluation  --g /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/experiment/MSD_liver/data/labelsTs --op /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/experiment/MSD_liver/result/se_U_fold_0/predict/0 --j /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/experiment/MSD_liver/result/se_U_fold_0/predict/0/comparison_results.json

python /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/Main.py --m postprocess  --ip /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/experiment/MSD_liver/result/se_U_fold_0/predict/0 --op /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/experiment/MSD_liver/result/se_U_fold_0/predict/post_0 --x /dssg/home/acct-clsyzs/clsyzs-zhyf/SAMG/experiment/MSD_liver/data/cancer_infor_cutpage_big_tumour_test.xlsx
