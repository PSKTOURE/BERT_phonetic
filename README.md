# BERT_phonetic

## Training
1. Install miniconda or anaconda
`mkdir -p ~/miniconda3`
`wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh`
`bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3`
`source ~/miniconda3/bin/activate`
`conda init --all`
`rm ~/miniconda3/miniconda.sh`

2. Create virtual env: 
`conda create --name bert python=3.10`
`conda activate bert`
3. Install requirements: `pip install -r requirements.txt`
4. Change hyperparameters in config.txt (optional)
5. The tokenizers has to be one of the listed in the tokenizers dir or from huggingface.
6. run main.py 
It will download the datasets and make the translation to phonetic if the datasets is not found, eats up a lot of space in disk 150Go+, 
and will also train using the default config found in config.txt
For training model: `python3 main.py --train`
For training tokenizer(optional): `python3 main.py --train_tokenizer`
For fine-tuning on all glue task: `python3 main.py --fine_tune`
Change args in config.txt if needed.