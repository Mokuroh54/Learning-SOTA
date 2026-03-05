# More Efficient Softmax Attention Mechanisms

Largely based off this godsend of a video: <https://www.youtube.com/watch?v=Y-o545eYjXM>. The creator Jiabin Huang makes some really amazing videos, and I have watched basically all of his videos from the past year for my learnings. Therefore, the structure will be very similar to the video, so it's basically a TLDR with some extra explanation on maybe implementation details that the video didn't get into or concepts that took me a bit longer to get.

Admittedly, drawing the line between what is attention mechanism A and attention mechanism B is pretty pointless. Instead, I want to mostly portray the innovations that get attention to where it is.

## Attention (2014)

In traditional attention, each token is associated with a query, key, and value vector. When processing token $t$, we compute $q_tK^T_{t-1}$, where $K_{t-1}$ contains the key vectors for each preceding token. We want the model to learn $q$ and $k$ such that these values indicate some sort of "importance score" for each previous token. In loose terms, $q_t$ represents the information the model is looking for, and $k_t$ represents the information that the token can provide. For example, if the model sees tokens "After a long day, my new car is still" and needs to predict a next word, we expect to see a high score for the token "new" but probably lower scores for "day". The attention scores are then multiplied with $V_{t-1}$, which contains the information that the token can give. Concretely, attention is written as:

$$
\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

where we use softmax to generate a probability distribution among the scores, and we scale by a factor of $\sqrt{d_k}$ to avoid vanishing gradients. The query, key, and value vectors are learned functions of the token embedding (i.e. $Q=XW_q,K=XW_k,V=XW_v$). Finally, the output is projected back into the embedding space.

## Multihead Attention (2017)

Having the model learn a single query and key vector that contains all the possible information about a token is hard. Instead, researchers thought of using multiple attention heads to distribute this burden. Now, instead of using $d_model$ dimensions to store a single query/key/value vector, we can store $h$ vectors of size $\frac{d_model}{h}$. Intuitively, even though the vector size is reduced by a factor of $h$, the information each distribution needs to handle is also reduced by $h$. Mathematically, the information that $h$ heads can cover is actually greater than a single head due to the nonlinearity of softmax. Therefore, it has been shown that MHA performs much better than traditional attention. In fact, MHA has been the last innovation to attention that is a cut and dry upgrade to performance.

In terms of implementation, everything up the the attention calculation can be left in-place, since we essentially just store the heads together. We do need to reshuffle a bit to make sure softmax operates on the right dimensions, but overall it is very straightforward.

### Interlude: KV Caching

So far, traditional attention and MHA have been significant advances in training, enabling more and more powerful transformers to be developed. Eventually, research turned to using transformers for generation, culminating in models like GPT-1.

One inherent problem is that generation will take $O(T^2E), since for every token we need to regenerate the $O(T)\$ previous tokens' kv vectors. Fortunately, this problem has an easy fix: once we calculate the key and value vectors, we can just store them for further use. Yippee! This allows short sequence generation to happen quickly, but what about longer sequences? Especially as more powerful models have greater qkv dimensions, the KV cache might grow to tens of gigabytes. Additionally, as memory grows larger, reading from memory becomes inefficient too.

Since training can take however long we like (ahem Deepseek v4), we will focus primarily on inference-time efficiency improvements from now on.

## Multi Query Attention (2019)

The whole idea of MHA was that we could describe different relation types with each head. However, do we really need to have $h$ sets of queries and $h$ sets of vectors at the same time? As an analogy, instead of forcing an astronomy student to use an astronomy textbook, a math student to use a math textbook, and so on, wouldn't it be enough for the astronomy student and math student to both use something like Wikipedia? In this way, we can generate $q$ heads but use only $1$ set of key and value vectors, reducing the cache size by a factor of $h$. Unfortunately, the answer is "sort of," since MQA has been shown to result in a nontrivial performance decrease.

Once again, the implementation changes are minimal. We just need to remember to duplicate the key vectors for attention. This can be done with `torch.repeat_interleave()`.

## Group Query Attention (2023)

Perhaps taking the number of kv heads down to $1$ was too much. What if we traded some efficiency back for the performance we lost? Therefore, GQA is a "middle ground," using $1 < h_k < h_q$ key heads. In practice, most of the accuracy lost with MQA was recovered with GQA, while still retaining a decent KV cache size decrease.

## Multihead Latent Attention (2024)

Admittedly, the next part seems somewhat unintuitive/hard to come up with. We can fill out the rest of the keys with `torch.repeat_interleave()`, but we claim that it makes more sense to express this operation as a block matrix multiplication. In a sense, we up-project the key and value matrices with a matrix $W_{ku}$ so that each query head sees one copy. It seems that we are using $O(T*d_{model})$ time to complete an operation that could just be done in $O(T)$, but this formulation allows us to consider using a down-projection matrix $W_d$ to achieve a low-rank factorization, which in turns makes the up-projection matrix nontrivial. We choose a latent dimension $d_c$ as the low rank.

Observe that because the keys and values are both projected into this low rank, we can use the same downprojected matrix (the video calls it $C_{kv}$, so I will too) for both the keys and values. In fact, during inference, we only need to save $C_{kv}$, which is much smaller than any previous kv cache we have seen! But why stop here? We can also low-rank factorize the query vectors for faster training. But most significantly, we realize that $W_{ku}$ no longer needs to enforce there being only $k_h$ distinct key heads. Instead, we can go back to MHA's setup of $h$ heads for keys and values, and we don't incur the cost of storing a large KV cache since we never materialize the keys or values directly.

But we can take this idea even further at inference time. Let's revisit our attention function all the way back.

$\text{softmax} (\frac{XQ_dQ_u^{(i)}{K_u^{(i)}}^TC_{kv}^T}{\sqrt{d_k}})C_{kv}V_u^{(i)}$

And this is just for a single head!

As $Q_uK_u^T$ is fixed, we can condense that into a single matrix. Note also that at the end of our calculation, we must right multiply our result by $W_o$, meaning we can also combine $V_uW_o$ together as well. All of these optimizations ensure that MLA can run much faster than any previous attention mechanism with similar performance to MHA.

### Rotary Positional Embedding

Traditional transformers use a sinusoidal positional embedding, which have several problems, but most of them center on the inherent fixedness of such a representation. It is difficult to generalize sinusoidal positional embeddings for high sequence lengths, and the embeddings are fixed on absolute indices of tokens, so patterns are hard to detect. Instead, researchers thought to use rotations around the origin as positional embeddings. This fixes the absolute nature of sinusoidal positional embeddings, and has largely replaced it in modern transformers. In practice, the embedding dimensions are split into two halves, and corresponding indices are paired to be rotated.

As good as MLA is, trying to use ROPE with MLA is difficult. Since the rotations depends on the exact query position, this prevents us from absorbing $Q_dQ_u^{(i)}K_u^{(i)}$, because we instead need to compute $Q_dQ_u^{(i)}R_qR_kK_u^{(i)}$. Since $R_q$ and $R_k$ change from token to token, preocmputing this is impossible. Instead, researchers decided to split the positional information from the content. In particular, we allocate $d_r$ dimensions to store the positional information of the queries and keys. 

The implmentation was a bit hard for me to grasp at first, particularly due to the asymmetry between how queries and keys are handled. But I think a good way to make this intuitive is to remember why attention still evolved after MHA: reducing KV cache. For the queries, we are free to generate one ROPE matrix per head, but since we want our KV cache to be as small as possible, and we only need one of ROPE matrix for the keys, that is our choice. This also explains why we generate the query ROPE matrix from the latent queries but the key ROPE matrix from our original tokens: our queries need to learn positional information per head, but because the keys are fixed, we can just use one representation. This method is called decoupled ROPE.

## Deepseek Sparse Attention (2025)

Sidenote: I wonder if there's some analogue to Moore's Law for context lengths.

Anyways the impetus behind DSA is that for very long sequences, we might attend to many tokens that might be irrelevant. For example, I have chats on Gemini where I ask questions about multiple topics in the same conversation, and it tries really hard to connect my current talking points with something we said 5 days ago, which is really not what I'm looking for. 

Instead of attending over all tokens, DSA proposes a lightning indexer that can generate a rough list of $k$ tokens that might be relevant, before the full attention is run on those $k$ tokens. The lightning indexer works similarly to attention, with some differences. Instead of using softmax, the indexer uses ReLU to zero out negative scores that don't provide useful information in the first place. Additionally, each head has a learnable weight that adjusts how much contribution they provide for every query. The weighted sum is computed to produce the final scores, and the top $k$ tokens are selected.

The obvious question here is "if lightning indexing is similar to attention, how does this actually help"? The key lies in the role of the indexer: a *coarse* approximation. This means we can cut some corners and tolerate a lower accuracy. First, like MQA, we can just create one set of key vectors instead of one for each head. Remember that the performance loss is acceptable here. More importantly, instead of using full precision, we can instead clamp our values down to fp8 precision, providing a 2x/4x speedup over fp16/fp32. This means that the full attention can run in $O(Tk)$ time, where $T$ is the sequence length, and while the lightning indexer still takes $O(T^2)$ time, the constant factor is improved. 

There is a potential issue though. We usually normalize the vector by dividing each value in the vector by the max, but if there some neurons that are significantly larger than others, clamping them to fp8 might cause them to "lose signal." For example, if they are off by a magnitude of 1000x, then the lower value will become $0$. To "smooth out" the vector, we apply the Fast Walsh-Hadamard Transform, which is a very optimized function on modern GPUs. That said, I think the use of the FWHT here is up for debate. I was also suggested to normalize the query and key vectors by the sqrt of their respective dimensions before feeding into the FWHT. It seems to make sense, but it's not in the paper and I don't know if it's the right choice.

To train this lightning indexer, we first allocate some number of warmup steps that allows the indexer to become a good approximator of the full attention. In particular, the loss becomes the KL divergence of the two distributions.

## Remarks

I don't think it's up for debate that attention is the most significant advancement to ML ever, and it's quite interesting to see how far it's come since its original proposition over a decade ago. It seems that the early research in improving attention was pioneered by Google (MHA, MQA, GQA), but lately Deepseek has been spending a lot of effort into optimizing Attention here. From what I know, compute is much harder to come by in China, which could be why the researchers working on Gemini haven't found a need for it yet. That said, the improvements become better as the models grow larger, so I'm interested in seeing what impact something like DSA would have on Gemini.

I had Claude Code write me a comparison script, but frankly the scale is nowhere large enough to effectively see the true improvements in time/performances each of the attention mechanisms have. DSA chooses the top 2048 tokens to do full attention on, and I will not be training on anything that makes the lightning indexer worth it at that scale. 

This is my first attempt at this, and frankly I'm still trying to figure out how to most effectively use Claude Code as a teacher and how much I should let it implement. I will say for sure I would not be doing this if Claude Code couldn't write a lot of the skeleton code for me. I do think that even though AI development is moving away from hand-writing code, writing out the key parts of the attention mechanisms by hand has taught me more (and made the concepts stick much easier) than just watching the video or reading papers/blogs. 

I don't know how to end this and I also don't know if people/who will read this. Hopefully they do. Thanks for reading, if anyone sees this :>