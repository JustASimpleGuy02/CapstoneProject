# RECOMMENDER SYSTEM USING FASHION CLIP

## Setup

``` bash
conda create -n fclip python=3.8
conda activate fclip
pip install -r requirements.txt
```

## Data preparation

### Original Polyvore

Data download from: [Here](https://www.kaggle.com/datasets/dnepozitek/polyvore-outfits)

Create a folder `data` in project directory

Extract the data to folder `polyvore_outfits`, place it in folder `data`

### Fashion Hash Net Polyvore

Data download from: [Here](https://stduestceducn-my.sharepoint.com/personal/zhilu_std_uestc_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fzhilu%5Fstd%5Fuestc%5Fedu%5Fcn%2FDocuments%2Fpolyvore&ga=1)

Extract the data to folder `polyvore`, place it in folder `data`

Extract the `291x291.tag.gz` file in `image` folder

### CSV files

Create a folder `csv_files`, place it in folder `data`

Download files from: [Here](https://drive.google.com/drive/folders/1EVNyUIoszvw4tNUrVsLgJ78FpHSNtZe_?usp=sharing)

### Demo files

Create a folder `demo`, place it in folder `data`

Download files from: [Here](https://drive.google.com/drive/folders/1EVNyUIoszvw4tNUrVsLgJ78FpHSNtZe_?usp=sharing)

Notes: the path in paths.txt is config for Windows, so please run the app in Windows.

## Colab

Get a copy from: [Here](https://colab.research.google.com/drive/13QyNX2XlQkaO42m7yktEaXr9fZKKlYHB?authuser=2#scrollTo=oACMlxLku3uE)
