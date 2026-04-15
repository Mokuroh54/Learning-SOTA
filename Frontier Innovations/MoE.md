# Mixture of Experts

## Background

Most of the attention given to transformers is with attention (pun half intended), but attention itself is just half the story. After attention, the new embeddings are fed through a small FFN. 

The FFN takes in the output of a RMSNorm layer, which learns some scaling factor for each dimension to prevent embedding vectors from growing arbitrarily large. Then, the vectors are up-projected into a hidden layer with a higher number of dimensions (this used to be $4$, modern SwiGLU experts use a ratio around $\frac{8}{3}$). In the projection matrix, we interpret each row as something that encodes a question about a token. Therefore, if the resulting activation has a high value, then it's likely the token is relevant to the question. Otherwise, it might be $0$ or negative. After applying a nonlinear activation function, we project back down to $d$ dimensions. Intuitively, we would love the hidden layer to be in higher dimensions, but this comes at obvious costs to time and memory consumption.

## MoE

The key insight is that each token will usually only activate a small amount of hidden layer activations. Therefore, we can divide the large FFN into multiple smaller FFNs, which we will call experts. When each token is being processed, we will only need to activate a subset of the experts to process the token. This means that we can have the knowledge of a large model while running at the speed of a much smaller one.

### The Router

Assigning tokens to experts can be done with a router that takes the form of a linear layer with output dimension equal to the number of experts. Then, the scores (after softmax) will dictate the relevancy of each expert for a token, and we can choose the top $k$ experts to send the token through. The scores will also help weight the outputs of each expert. More formally, we compute $\sum_{i \in \mathcal{T}} p_i(x) E_i(x)$.

### Shared Expert

Deepseek v2 showed that having a single shared, always on expert is a good idea, and intuitively this makes sense. There are usually some traits that all tokens might have in common, such as part of speech, modifiers for time/relative comparisons, etc. This allows the selectively activated experts to become more specialized, while the shared expert carries the brunt of the other work. 

## Problem 1: Load Balancing

One problem with the MoE architecture we have built so far is load balancing. Because routing is decided dynamically, it may be the case that some experts get more tokens than others. This means that some experts will take longer than others to finish, leading to wasted compute.

For now, let's consider forcing each expert to process the same number of tokens. Consider the following paradigm where we denote $X_i$ as the matrix which contains the tokens assigned to the $i$ th expert, and $W_i$ as the up-projection matrix of the $i$ th expert. Instead of computing $X_iW_i$ individually, we can organize them as block diagonal matrices within a GPU. Of course, we don't want to materialize the zeros, so we can use sparse block diagonal matrix multiplication. And in fact, our restriction of using a fixed number of tokens per expert can be relaxed. Regardless of whether we divide the tokens evenly, or give every token to 1 expert, we still do one matrix multiplication per GPU.

## Problem 2: Training

Load balancing also has an effect on training dynamics. If some experts get more tokens than others, this means more gradients pass through them, and they become stronger while the other experts remain weaker. Over time, we end up with only a few useful experts and many dead ones.

One first try is to calculate the sum of probabilities assigned to the expert by the router, which we call importance. Then, we can calculate the coefficient of variance of importance scores, and try to minimize this value. But obviously, it is very easy to design counterexamples. Consider the case where we only select the highest probability expert out of $4$ and we have $4$ tokens. Say expert $1$ gets a probability of $0.5$ for tokens $1$ and $3$, expert $2$ gets a probability of $0.5$ for tokens $2$ and $4$, and experts $3$ and $4$ get probability $0.25$ for all tokens. Even though the importance score sum is the same, only experts $1$ and $2$ are ever activated.


To fix this, we will count for each expert $i$ the number of tokens it receives. To allow gradients to flow, we approximate this value with a continuous function. Now, we can design a load balancing loss of the form

$$L_{lb}=w_{importance}CV(importance)^2+w_{load}CV(load)^2$$

which we can simplify to 

$$L_{lb}=\alpha N \sum_{i=1}^N f_ip_i$$

where N is the number of experts, $f_i$ is the fraction of tokens assigned to expert $i$, $p_i$ is the fraction of probabilities assigned to expert $i$, and $\alpha$ modulates the contribution of the load balancing loss to the true objective. We scale the load balancing loss by $N$ to ensure that $N \sum_{i=1}^N f_ip_i \geq 1$. A similar approach is also viable at the device-level.

### Problem 2.5: How many losses do you want?

Ideally we don't want to have to assign a loss here, since that could muddy the training objective. In response to this, Deepseek proposed the following solution: instead of modulating probabilities with softmax, we can replace it with a sigmoid and use a bias vector that pulls the load of each expert closer to the mean. However, training stability using loss-free load balancing has been questionable.

## Problem 3: Stability

If we compute softmax over logits, we can easily see that adding a constant to each term does not affect the final result. Therefore, if we need to exponentiate each logit, the resulting value can blow up. Fortunately, safe softmax can negate this issue, but we would still rather just have smaller logits to begin with. We can add a loss to penalize large logits with a sum of squares, but this might hurt model performance since it interferes with the softmax probabilities. Instead, we can penalize the square of the log of the denominator of the softmax. This is the router Z-loss. 

## SOTA (for now)

 For load balancing, Qwen3 introduced a global batch load balancing loss, which computes balance across the macro batch instead of the micro batch within a GPU. Furthermore, there is still experimentation on whether to alternate MoE/Dense, whether to use shared experts, and how MoE can synergize with linear/softmax attentions.