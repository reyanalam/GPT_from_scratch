# GPT_from_scratch

## Step 1: Tokenization

GPT follows a subword based tokenization. It utilizes Byte-Pair Encoding.
Byte-Pair Encoding is a technique of word compression. The main idea behind the it is to break the rare words into subword and leave the more frequent word as it is.

### Why not doing word based tokenization or character based tokenization?

1) Word based tokenization can have an issue of out of vocabulory words(OOV). If a word that is outside the vocabulory is provided, the result will be an error.

To deal with it we can use '<|unk|>' as out of vocabulory words and assign them last token value. Another issue which may arise is the memory of the vocabulory. There are over 2 millions words. This will be memory inefficient.

3) Character based tokenization can deal with the issue that is faced by word based tokenization. However the main problem with character based tokens is that the meaning of word is entirely lost.

For these reasons sub-word based tokenization is prefered.

### How to do subword based tokenization?

It can be easily done by a python library, Tiktoken. Tiktoken is a fast and efficient tokenization library developed by OpenAI.

## Step 2: Data Sampling through data loader and input output pair

### Input-Output Pair

GPT is an autoregressive model, i.e., output of previous step is used for future prediction. The input is a 2D tensor of size (token IDs, maximum number of tokens) containing batchs of inputs and the output is a 2D tensor of same length contating the token IDs shifted by one.
This input output pair serves as data for self supervised learning.

### Data Loader

We use Dataloader and Dataset from data.utils class of Pytorch for data sampling. The input output pair is created and sampled through this class.

## Step 3: Token Embeddings

Eaxh token ID is converted a vector embedding. Vectors can capture semantic meaning, therefore it becomes an important step.

### How are token embeddigns created for LLMs?

The embedding weights are assigned random values which are later optimized as part of the LLM training process.

I have used Embedding layer from torch.nn.Embeddings to create a lookup table. It is called a lookup table because with respect to corrosponding token ID a vector of that index is found in the matrix.

The dimension of this lookup table is vocabulory_size * embedding_dimension. 

## Step 4: Position Embeddings

Transformers losses position of the tokens since they process each tokens parallely. To deal with it we add a positional embedding vector to the token embedding vector. The aim of this step is to capture the relative position of the tokens.

The size of positional embedding vector is context length, embedding_dimension. 

## Step 5: Droupout

The addition of token embeddings and positional embeddings results in our input embedding. This input embedding is fed into a dropout layer. The aim of the droupout layer is to prevent overfitting and to deal with lazy neurons problem.

## Step 6: Transformer Block

### Layer Normalization

The embeddings of each token is normalized such that there mean is equal to zero and their variance is equal to 1. This aim of this is to prevent internal covairant shift and to ensure stable training.

The layer normalization is done by subtracting each value of the embedding by their mean and is then dividied by square root of varaince.  

### Masked Multi-Head Attention

This is the main step of the transformer block that is responsible for creating contextual vectors of the token. The difference between contexual vectors and input embeddings is that the contexual vectors are more richer in terms of information as compared to the input embeddings. It not only captures the semnatic meaning like input embeddings but also captures the relation between the tokens themselves.

Multi-Head masked attention is implemented by a mutiheadattention class. The class takes input and output dimension, context length and number of heads as input. It creates a query, key and value weight matrix, the values of which will be optimized while training. These weight matrices when multiplied by input embeddings created query, key and value matrices, each is 3D tensor of shape 
(number_of_batches, number_of_tokens, dimension_of_output). 

These 3D tensors are converted into 4D tensors by defining a new parameter head_dimension, which is equal to output_dimension/number_of_heads. The shape of the 4D tensor is  
(number_of_batches, number_of_tokens, number_of_heads, head_dimension).

This 4D tensor is then transposed with respect to (1,2) i.e., the shape of query, key and value tensors is (number_of_batches, number_of_heads, number_of_tokens, head_dimension).

Now the attention score is calculated by matrix multiplying query and key transpose with respect to (2,3). This results is attention score tensor having dimension of  
(number_of_batches, number_of_heads, number_of_tokens, number_of_tokens).

GPT 2 uses causal attention i.e., the attention score of a token depends upon the tokens which comes before it. Therefore all the token in front of it is masked. 

Attention weights are then computed by normalizing the attention score by applying a softmax function. The attention weight is then multiplied by value tensor to compute the context vectors. The shape of whoes is (number_of_batches, number_of_heads, number_of_tokens, head_dimension). This 4D tensor is then transposed with respect to (1,2) resulting in tensor of shape 
(number_of_batches, number_of_tokens, number_of_heads, head_dimension).

Now each token is flattend into each row resulting in 3D tensor of shape (number_of_batches, number_of_tokens, dimension_of_output). This shape is same as that of our input to multihead attention class.

### Dropout

The output of multihead attention is fed into droupout layer.

### Shortcut connection

A shortcut connection is added to our vectors. This is done to create an alternate path for flow of the gradient. While backpropagation the gradient is calculated. This flow of gradient can create vanishing gradient problem. This is delt by creating an alternate path of flow of the gradient.

### Layer Normalization

The output is normalized such that the mean of each vector becomes zero and the variance becomes one.

### Feed Forward Neural Network

It is a neural network that deals with each token separately. The first layer increases the embedding_dimension 4 times. The second layer decreases it by 4 times. In between GELU activation function is also applied. The aim of this is to capture more feature from each token. 

The reason for GELU activation function to be used instead of others, is that the function is differenetiable throughout and is not zero for the negative values. Thus deals with the problem of dead neuron.  

### Dropout

The output of feed forward neural netwrok is fed into droupout layer.

### Shortcut connection

A shortcut connection is added to our vectors

## Step 7: Layer Normalization 

The output is normalized such that the mean of each vector becomes zero and the variance becomes one.

## Step 8: Output head

It is a neural network that converts our tokens vector into a vector of size equal to vocabulory size. Thus each token is represented in terms of all the tokens in the vocabulory.


