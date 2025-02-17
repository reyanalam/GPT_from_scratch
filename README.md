# GPT_from_scratch

## Step 1: Tokenization

GPT follows a subword based tokenization. It utilizes Byte-Pair Encoding.
Byte-Pair Encoding is a technique of word compression. The main idea behind the it is to break the rare words into subword and leave the more frequent word as it is.

### Why not doing word based tokenization or character based tokenization?

**1) Word based tokenization can have an issue of out of vocabulory words(OOV). If a word that is outside the vocabulory is provided, the result will be an error. 
To deal with it we can use '<|unk|>' as out of vocabulory words and assign them last token value. Another issue which may arise is the memory of the vocabulory. There are over 2 millions words. This will be memory inefficient.
**2) Character based tokenization can deal with the issue that is faced by word based tokenization. However the main problem with character based tokens is that the meaning of word is entirely lost.

For these reasons sub-word based tokenization is prefered.

### How to do subword based tokenization?

It can be easily done by a python library, Tiktoken. Tiktoken is a fast and efficient tokenization library developed by OpenAI.

## Step 2: Data Sampling through data loader and input output pair

### Input-Output Pair

GPT is an autoregressive model, i.e., output of previous step is used for future prediction. The inpput is a 2D tensor that contains number of token IDs and the output is a 2D tensor of same length contating in the token IDs, but with a stride. The output is slided ahead of the input token by a number equal to stride length.
This input output pair serves as data for self supervised learning.

### Data Loader

We use Dataloader and Dataset from data.utils class of Pytorch for data sampling. The input output pair is created and sampled through this class.

## Step 3: Token Embeddings



