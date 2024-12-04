from src.predict_last_verse_word import predict

dataset_path = "/home/toure215/BERT_phonetic/DATASETS/rap/phonetic_rap_ds_hf"
model_path = "psktoure/BERT_BPE_phonetic_bookcorpus_0.1"

predict(
    dataset_path=dataset_path,
    model_path=model_path,
    num_epochs=3,
    batch_size=32,
    num_iterations=5,
    k=5,
    log_file='rap_predict_last_word.tsv'
)
