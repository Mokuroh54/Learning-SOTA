# Attention Residuals

The goat is back: https://www.youtube.com/watch?v=LSHTkbnmzy4

## Background

Residual connections are a fundamental ML concept that predates even transformers and attention. They were invented to help solve the problem of vanishing gradients as the first deep learning models emerged. 

Suppose a model has input $x$ and layers $f_1, f_2, \cdots f_n$. Each hidden state would be calculated by applying the layer's function to its inputs ($x_1=f_1(x_0), x_2=f_2(x_1), \cdots x_n=f_n(x_{n-1})$). However, consider the gradients flowing through each layer. By the chain rule, we see that 

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\partial \mathcal{L}}{\partial x_n} \prod_{j=i+1}^n \frac{\partial x_j}{\partial x_{j-1}}$$

Near the optimum, where individual gradients are low, this product tends to collapse towards $0$.

Residual connections help alleviate the problem by adding the input of a prior layer to its input. In transformers, this is usually the input of the layer immediately before, whereas in other architectures it differs. Since we're focusing on transformers, we can rewrite our layer function as $x_i=x_{i-1}+f(x_{i-1})$. Accordingly, our gradient becomes

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\partial \mathcal{L}}{\partial x_n} \prod_{j=i+1}^n (\frac{\partial x_j}{\partial x_{j-1}} + I)$$

This means there is always at least one pathway that is the identity, so there is some signal that goes through.

## Full Attention Residuals

Implicitly, the weights of the residual connection as well as the layer are both $1$. However, we might want to be able to learn these weights, and we can use attention! For each layer, we learn a query vector $q_i$ and use $x_{i-1}$ and $f(x_{i-1})$ as key vectors for our attention score calculation. Note that while in normal attention we learn new matrices $W_q$ and $W_k$, here we treat the $q_i$ vector as a weight in the larger network initialized to $0$, and we directly use $x_{i-1}$ and $f(x_{i-1})$ as the key and value vectors. This ensures that the model starts out with equal weighting. Note that because the magnitudes of values in different layers might be different, we want to normalize the keys with RMSNorm. In practice, this leads to better performance and more stable gradients and activation magnitudes.

However we have 2 key problems. First: gradient checkpointing will fail. The idea of gradient checkpointing is that instead of storing all the activations, we can save only a portion of them and recompute the activations for the other layers. To put this in comparison, we need $O(N)$ storage for the model and $O(1)$ to retrieve gradient without gradient checkpointing. With gradient checkpointing with every $\sqrt{L}$ layers, we only need $O(\frac{N}{\sqrt{L}})$ memory at the cost of $O(\sqrt{L})$ computation for the gradient. 

Second: Say it with me, sequential techniques are not hardware friendly.

## Block Attention Residuals

Fortunately, it is not hard to design a solution. Since we alreadly separate the layers into blocks based on checkpoints, why don't we do the same for residuals? Instead of storing $\sqrt{L}$ query weights per block, just store a block query vector that is created by summing the individual query vectors. The nice part is that while we are constructing a block, using its own block query vector still gives the correct attention computation. Intuitively, assigning each block to a GPU is a good idea.

We haven't fully solved the sequential problem though. At the very least, each GPU spends time waiting for the other GPUs to finish their computation before it is used again. We can partially fix this issue by dividing along the batch dimension. If for example we have $4$ GPUs, we can divide the batch into $4$ quarters. Therefore, after GPU 1 is done with computing the first block of the first quarter, it can immediately move on to the second quarter while GPU2 computes the second block of the first quarter. 

The other way we can increase computation speed is by thinking about how we can transfer information between GPUs. To do this, we introduce some number of virtual stages, where each GPU computes some layers in a virtual stage. Assuming a 4 GPU setup, let's say in the first virtual stage, GPU 1 computes 2 layers of the 4-layer block 1. The next time GPU 1 needs to do some computation will be the first 2 layers of block 3. We can't avoid transfering the block summaries for layers 2-6, but the block summary for the first 2 layers is already on our GPU! This might not seem like a big improvement, but the time saved scales with the number of virtual stages.

Finally, we can make one small optimization to the attention computation among blocks. We can precompute the attention scores for the current block to previous blocks, and use online softmax (as in FlashAttention) to compute the intra-layer attention scores.