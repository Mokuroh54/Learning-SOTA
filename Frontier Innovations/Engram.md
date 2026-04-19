# Deepseek Engram

## Background

In LLMs, we follow attention layers with FFNs and/or MoE layers. These layers will up-project the new embeddings into a higher dimension and then down-project them back into the original embedding dimension. This can be thought of as a reorganization of information stored by the LLM. More specifically, we can express the FFN as 

$$FFN(x)=W_2 \sigma (W_1x)$$

, and in this formulation we can think of each row of $W_1$ as extracting a fact about $x$. The most relevant responses are filtered with $\sigma$. Then, each column of $W_2$ can be thought of encoding some important information, which the importance scores given by $\sigma(W_1x)$ will filter for.

In both these scenarios, we extracted knowledge by computation, but what if we can extract knowledge by lookup?

And no I'm not talking about RAG.

## Lookup Table

To create our lookup table, we can start by mapping the vocabulary to embeddings that store facts, but sometimes, we might want to map a bigram of tokens to a fact. Therefore, we use a hash function to store the index of facts. To avoid collisions, we use the hash function $m_0x_0 \oplus m_1x_1 \mod M$, where $m_0$ and $m_1$ are odd to preserve the unit bit and $M$ is some large prime. Using two multipliers has the added benefit that the hash becomes order-aware, so "dog hot" and "hot dog" will not extract the same facts. This method generalizes to a n-gram of higher number of tokens. We can also apply multiple hashes with different parameters to further reduce collisions.

## Engram

However, it turns out that most of the embeddings retrieved by Engram encode patterns instead of semantics. Therefore, we apply a context-aware gate on these embeddings as follows:

$$\tilde{v}=\sigma(\frac{\text{RMSNorm}(x) \cdot \text{RMSNorm}(eW_{ke})}{\sqrt{d}}) * (eW_{ve})$$

where $x$ is our token hidden state, $e$ is the retrieved embedding from Engram, $W_{ke}$ and $W_{ve}$ are separate key and value matrices respectively, $d$ is the dimension of the vectors, and $\sigma$ is a sigmoid gate.

Finally, we output

$$y=\tilde{v} + \text{SiLU}(\text{Conv1D}(\text{RMSNorm}(\tilde{v})))$$

We choose to use a depthwise 1D Conv to widen the recptive field and combine it with SiLU for nonlinearity. We reinject the $y$ into each token's hidden state. 

## Implementing Engram

Deepseek's experiments reveal that the optimal distribution of Engram and MoE lies at about using 25% of compute on Engram and 75% on MoE. In particular, placing Engram layers early helps take off the burden for MoE layers to reconstruct facts, allowing them to focus more on reasoning. Just note that placing it on the first layer is suboptimal, as the hidden states for the tokens have not accrued meaning yet. Furthermore, since the embedding table is unchanging during inference, we can offload it to host memory, freeing up memory at inference time for the GPU to perform active computations with.