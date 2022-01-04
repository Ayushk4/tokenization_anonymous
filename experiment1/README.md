# Experiment 1: Do Language Models know their characters

## Dependencies

- [nltk >= 3.6](https://www.nltk.org/install.html)
- [PyTorch >= 1.7.1](https://pytorch.org/get-started/previous-versions/)
- [Transformers > 4.10](https://huggingface.co/docs/transformers/installation)
- [Sentencepiece >= 0.1.91](https://pypi.org/project/sentencepiece/)

NLTK requires additional data dependencies.
```
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

## Setting up

1. Download [GPT-J's Embeddings](https://github.com/Anonymous-ARR/Releases/releases/download/gptj/gpt-j-6B.Embedding.pth) into the current folder with name `gpt-j-6B.Embedding.pth`.
2. Run the experiments for GPT-J using `python3 train.py --seed=[SEED] --batch_size=[BATCH_SIZE] --lr=[LEARNING_RATE] --n_epochs [NUM_EPOCHS] --wandb`.
3. For control experiments include the additional flag `--control`.
4. For Non-GPT-J based experiments include the additional parameter `--control=EleutherAI/gpt-j-6B`.
