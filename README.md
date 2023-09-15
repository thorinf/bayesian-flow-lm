# bayesian-flow-lm

Applying [Bayesian Flow][1] to Language Modeling in Pytorch.

## How to Run

### Environment Setup

Aside from PyTorch, the training script requires three other packages;
<a href="https://github.com/google/sentencepiece">sentencepiece</a>, 
<a href="https://github.com/lucidrains/rotary-embedding-torch">rotary-embedding-torch</a>, and
<a href="https://github.com/thorinf/bayesian-flow-pytorch">bayesian-flow-pytorch</a>.

They can be installed with the following commands:

```commandline
pip install sentencepiece
pip install rotary-embedding-torch
pip install git+https://github.com/thorinf/bayesian-flow-pytorch
```

### Creating a Tokenizer

To use a SentencePiece tokenizer model, you have two options:

1. Create a new SentencePiece tokenizer model.
2. Find and use an existing one. 
 
Here are the steps to create a new model:

First generate a `.txt` corpus where each line is an example.
It's recommended to apply some normalisation on the text so the data is quite clean for the next step and training, e.g.
lower-case, change numbers to words, removing unnecessary symbols.
The training script won't perform these normalisations, so data should be cleaned externally.

With a clean text corpus, the SentencePiece model can then be trained.
Follow the guides on [their repository](https://github.com/google/sentencepiece)
or [here on PyPI](https://pypi.org/project/sentencepiece/).
If the text corpus is very large, then creating a subset of the text can get around memory issues.
Here is an exert from the script that created the BPE model: 

```python
spm.SentencePieceTrainer.train(
    input=text_path,
    model_prefix=name,
    model_type='bpe',
    vocab_size=size,
    user_defined_symbols=[str(i) for i in range(10)],
    bos_id=0,
    eos_id=1,
    pad_id=2,
    unk_id=3
)
```

### Training

The model can be trained with the command:

```commandline
mkdir MODEL_DIRECTORY
python train.py -d=TXT_CORPUS -spm=SPM_MODEL -mdir=MODEL_DIRECTORY
```

## Design & Implementation Notes

### Architecture

There is currently no implementations for Bayesian Flow Networks at this scale. 
In the [paper][1], Graves et al. train a language model but only for graphemes, which is a small vocabulary.
However, there are other successful generative language models, outside of Bayesian Flow Networks, 
that can be used as inspiration.

From [CDCD][2] (Dieleman et al.) we can mirror the Transformer architecture. 
This is an 8 Layer, 8 Head Transformer, using [FiLM][3] for applying the time-step embedding to each encoder layer.
This architecture is designed to be applied to noised embeddings, 
but in [Bayesian Flow Networks][1] the input to the model is a distribution.
So solve this the Transformer model uses the distribution to take a weighted sum of the embedding weights, 
this is the approach taken in [SSD-LM][4].

For conditioning positions, a one-hot vector is created to update the input distribution. 
This is only applied to positions where the conditioning mask is `True`. 
Taking the weighted sum of a one-hot vector with the embedding weights results in just the embedding for the correct ID.
To inform the model that this is a conditional embedding, the time-steps for conditioniing positions are set to 1.
During generation a time-step of 0 should indicate that the distribution is completely inaccurate, 
but a time-step of 1 should indicate that the distribution is accurate.
By setting the conditioning positions to have a time-step of 1, we have told the model that the prediction is final.

### Experimentation

As Bayesian Flow Networks are a new class of Generative Model and there is little literature on the topic,
selecting hyperparameters for training is perhaps the more difficult part.
Specifically, selecting `beta` is very important.
There appears to be an inverse relationship between `num_classes` and `beta`.
If `beta` is too high given the `num_classes` then the output of `theta`
will often match the ground truth distribution.
Since there is little to no corruption of the data, there is little for the model to learn.
In these cases of a large `beta`, the loss may start high because of loss weight scaling, 
but fall to near zero after relatively few weight updates.
However, if `beta` is set lower, then the loss may start lower because of scaling, 
but the loss will not instantly drop.
Exactly how to select `beta` isn't clear, but picking something sufficiently small to corrupt the data is necessary.

## Citations

```bibtex
@misc{graves2023bayesian,
      title={Bayesian Flow Networks}, 
      author={Alex Graves and Rupesh Kumar Srivastava and Timothy Atkinson and Faustino Gomez},
      year={2023},
      eprint={2308.07037},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
[1]: <https://arxiv.org/abs/2308.07037>
"Bayesian Flow Networks
"

```bibtex
@misc{dieleman2022continuous,
      title={Continuous diffusion for categorical data}, 
      author={Sander Dieleman and Laurent Sartran and Arman Roshannai and Nikolay Savinov and Yaroslav Ganin and Pierre H. Richemond and Arnaud Doucet and Robin Strudel and Chris Dyer and Conor Durkan and Curtis Hawthorne and Rémi Leblond and Will Grathwohl and Jonas Adler},
      year={2022},
      eprint={2211.15089},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

[2]: <https://arxiv.org/abs/2211.15089> 
"Continuous diffusion for categorical data"


```bibtext
@misc{perez2017film,
      title={FiLM: Visual Reasoning with a General Conditioning Layer}, 
      author={Ethan Perez and Florian Strub and Harm de Vries and Vincent Dumoulin and Aaron Courville},
      year={2017},
      eprint={1709.07871},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

[3]: <https://arxiv.org/abs/1709.07871>
"FiLM: Visual Reasoning with a General Conditioning Layer"

```bibtex
@misc{han2023ssdlm,
      title={SSD-LM: Semi-autoregressive Simplex-based Diffusion Language Model for Text Generation and Modular Control}, 
      author={Xiaochuang Han and Sachin Kumar and Yulia Tsvetkov},
      year={2023},
      eprint={2210.17432},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

[4]: <https://arxiv.org/abs/2210.17432> 
"SSD-LM: Semi-autoregressive Simplex-based Diffusion Language Model for Text Generation and Modular Control"




