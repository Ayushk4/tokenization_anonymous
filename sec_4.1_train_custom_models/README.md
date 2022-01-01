# Sec 4.1 Training Custom NER/POS Models for Predicting Characters

## Dependencies

- [PyTorch >= 1.7.1](https://pytorch.org/get-started/previous-versions/)
- [Transformers > 4.10](https://huggingface.co/docs/transformers/installation)
- [Sentencepiece >= 0.1.91](https://pypi.org/project/sentencepiece/)
- [Scikit-Learn >= 0.24.2](https://scikit-learn.org/stable/install.html)
- [Wandb >= 0.12.2](https://docs.wandb.ai/quickstart#1.-set-up-wandb)


## Setting up
 
1. Install above dependencies, you may also install cudatoolkit if needed.
1. Download [GPT-J's Embeddings](https://github.com/Anonymous-ARR/Releases/releases/download/gptj/gpt-j-6B.Embedding.pth) into the current folder.

You may train the model using NER and PoS from these custom pretrained models using - `python3 train.py --seed=[SEED] --batch_size=[BATCH_SIZE] --lr=[LR] --n_epochs=[N_EPOCHS] --bert_type=[BERT_TYPE] --task={pos,ner} --task_level={sentence,token} --wandb`. The Wandb command is optional and using it will allow you to use WANDB for logging.

`bert_type` should be among those models supported by huggingface. We used `EleutherAI/gpt-j-6B` and `bert-based-cased`

For example for the gptj model you can use:
`python3 train.py --task ner --bert=EleutherAI/gpt-j-6B --task_level token --batch 64 --lr 5e-5 --n_ep 20`

For the GPT-J model, the task_level token and sentence will train the same model.

If you are using `cpu`, set also include `--device=cpu`.

It takes about 2 minutes to convert the CONLL dataset for POS and NER Tagging. The is cached and runs fast for subsequent usage.

It takes about 20 minutes for the Bert-Model and 10 minutes for MLP based model over GPT-J.

It is recommended to clear cache each time you change the dataset. As it may use the previous models' cache instead of the new one.


## Dataset Credits:

The CONLL 2003 formatted dataset was obtained from https://github.com/davidsbatista/NER-datasets and is curated by [Sang et al. 2003](https://aclanthology.org/W03-0419/)
