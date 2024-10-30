# BERT_phonetic

## Training
1. Install miniconda or anaconda
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
source ~/miniconda3/bin/activate
conda init --all
rm ~/miniconda3/miniconda.sh
```

2. Create virtual env:
```bash
conda create --name bert python=3.10`
conda activate bert
```
3. Install requirements: `pip install -r requirements.txt`
4. Install lex_lookup (necessary for epitran):\
```bash
git clone https://github.com/festvox/flite.git
cd flite
./configure --prefix=/home/.local/bin #(if you don't have sudo right add the prefix part)\
make
make install
make lex_lookup
sudo cp lex_lookup /usr/local/bin # if sudo right else 
cp lex_lookup /home/.local/bin
```
Make sure that /home/.local/bin is in your $PATH. 
5. Change hyperparameters in config.txt (optional)
6. The tokenizers has to be one of the listed in the tokenizers dir or from huggingface.
7. Set HunggingFace access token (optional):\
Grab your access token from your huggingface account and add
```bash
echo export HF_TOKEN="your_secret_token" >> .bashrc
source .bashrc # or .zshrc depending of your shell
conda activate bert
```
If set will upload model to you hub account else will save
it in the model directory.
8. run main.py 
9. Download bookcorpus dataset or wikitext or glue for testing.
```python
python3 main.py --download_bookcorpus # it will eat up a lot of space 250Go+
python3 main.py --download_wikitext
python3 main.py --download_glue
```
For training model: `python3 main.py --train`\
For training tokenizer(optional): `python3 main.py --train_tokenizer`\
For fine-tuning on all glue task: `python3 main.py --fine_tune`\
Change args in config.txt if needed.
10. Trained model will be in the models directory
11. Clean your .cache directory after training to regain around 250Go of space.