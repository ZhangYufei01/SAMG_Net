# SAMG-Net: Leveraging Semi-Automatic Mask Guidance for Enhanced 3D Tumor Segmentation in Medical Imaging

This project contains code for performing a specific operation. Below are the steps for calling the code:

## Installation

To install the required libraries for this project, please refer to the `requirements.txt` file.

## Steps

1. Set the Python path:

    ```bash
    export PYTHONPATH="your_path/SAMG:$PYTHONPATH"
    ```

2. Preprocessing:

    ```bash
    python your_path/SAMG/Main.py --m preprocess --pre your_path/SAMG/experiment/{your_task}/data
    ```

3. Train the model:

    ```bash
    python your_path/SAMG/Main.py --m train --d your_path/SAMG/experiment/{your_task}/data/fold_0.json --o your_path/SAMG/experiment/{your_task}/result/se_U_fold_0 --pd your_path/SAMG/experiment/{your_task}/data/preprocess_data --f 0 --p "(64,192,192)"
    ```

4. Model prediction:

    ```bash
    python your_path/SAMG/Main.py --m predict --d your_path/SAMG/experiment/{your_task}/data/predict_semi.json --o your_path/SAMG/experiment/{your_task}/result/se_U_fold_0/predict --f 0 --c your_path/SAMG/experiment/{your_task}/result/se_U_fold_0/checkpoint_best.pth --p "(64,192,192)"
    ```

5. Model evaluation:

    ```bash
    python your_path/SAMG/Main.py --m evaluation --g your_path/SAMG/experiment/{your_task}/data/labelsTs --op your_path/SAMG/experiment/{your_task}/result/se_U_fold_0/predict/0 --j your_path/SAMG/experiment/{your_task}/result/se_U_fold_0/predict/0/comparison_results.json
    ```

6. Post-processing:

    ```bash
    python your_path/SAMG/Main.py --m postprocess --ip your_path/SAMG/experiment/{your_task}/result/se_U_fold_0/predict/0 --op your_path/SAMG/experiment/{your_task}/result/se_U_fold_0/predict/post_0 --x your_path/SAMG/experiment/{your_task}/data/cancer_infor_cutpage_big_tumour_test.xlsx
    ```

**Note:** 
- Replace `your_path` with the actual path to your project directory.
- Customize `{your_task}` to match your specific task within the project.
- You can adjust the patch size parameter `"(64,192,192)"` according to your configuration requirements.
