# FASHION RECOMMENDER SYSTEM BY PROMPS

## Setup
```
$ conda create -n fclip python=3.8
$ conda activate fclip
$ conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia
$ pip install -r requirements.txt
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

```
$ unzip ${csv_file} -d data/
```

### Demo files

Create a folder `demo`, place it in folder `data`

Download files from: [Here](https://drive.google.com/drive/folders/1EVNyUIoszvw4tNUrVsLgJ78FpHSNtZe_?usp=sharing)

```
$ unzip ${demo_file} -d data/
```

## Colab

Get a copy from: [Here](https://colab.research.google.com/drive/13QyNX2XlQkaO42m7yktEaXr9fZKKlYHB?authuser=2#scrollTo=oACMlxLku3uE)

## Training
```
$ ./train.sh
```

## Deployment


## For testing only
```
$ pip install fashion-clip
```

## Reference

``` text
@misc{cg2021trainCLIP,
  author = {Cade Gordon},
  title = {train-CLIP},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4915843},
  howpublished = {\url{https://github.com/Zasder3/train-CLIP}}
}
```
