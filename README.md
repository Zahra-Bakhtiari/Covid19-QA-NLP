# Covid19-QA-NLP

Prerequisites:

* Download BERT Base, Uncased model to a local directory (e.g. bert_base_uncased)
* Download SQuAD 1.0 dataset to a local directory (e.g. squad1)
* Create a new virtual envirnoment
* Install python version 3.7 or lower. Note 3.8: version is not compatible with features in this github 
* Install Tensor flow version 1.0. TF1.15 is recommanded.

Baseline: Fine-tune the BERT model with Squad QA dataset, then test it on the COVID QA dataset.
```
python3 bert/run_squad.py \
  --vocab_file=../bert_base_uncased/vocab.txt \
  --bert_config_file=../bert_base_uncased/bert_config.json \
  --init_checkpoint=../bert_base_uncased/bert_model.ckpt \
  --do_train=True \
  --train_file=../data/squad1/train-v1.1.json \
  --do_predict=True \
  --predict_file=../data/question-answering/COVID-QA-dev.json \
  --train_batch_size=16 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=256 \
  --doc_stride=128 \
  --output_dir=../covid_nlp_output/
```

Evaluate the performance of the baseline.
```
python3 bert/evaluate.py ../data/question-answering/COVID-QA-test.json ../covid_nlp_output/predictions.json
```
Evaluation metrics.

Dataset | Exact Match | F1
------------|------------|------------
dev 28.46 | 48.08

Fine-tune the BERT model with shuffled squad and covid mixed training dataset, then test it on the COVID QA test dataset.
```
python3 bert/run_squad.py \
  --vocab_file=../bert_base_uncased/vocab.txt \
  --bert_config_file=../bert_base_uncased/bert_config.json \
  --init_checkpoint=../bert_base_uncased/bert_model.ckpt \
  --do_train=True \
  --train_file=../data/question-answering/squad-covid-combined-training.json \
  --do_predict=True \
  --predict_file=../data/question-answering/COVID-QA-dev.json \
  --train_batch_size=16 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=256 \
  --doc_stride=128 \
  --output_dir=../covid_nlp_output2/
```

Evaluate the performance of the baseline.
```
python3 bert/evaluate.py ../data/question-answering/COVID-QA-test.json ../covid_nlp_output2/predictions.json
```
Evaluation metrics.

Fine-tune the uncased BERT model having L-12, H-768, A-12  with `squad-covid-combined-training.json`.
Dataset | Exact Match | F1
------------|------------|------------
dev | 31.43 | 57.33


Fine-tune BioBERT model with `squad-covid-combined-training.json`.

Dataset | Exact Match | F1
------------|------------|------------
dev | 25.94 | 46.27


