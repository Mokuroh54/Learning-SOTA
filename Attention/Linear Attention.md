# Linear Attention

This assumes prior reading/knowledge of Softmax Attentions.md. If you haven't read it yet, a brief skim might help refresh knowledge on attention + contextualize where we are.

We've talked a lot about optimizing attention by using different numbers of heads, factorizing into low ranks, and even using a coarse approximator with a lightning indexer. But what if we could fundamentally improve attention itself? Softmax is a pretty heavy operation, and we can't use tensor cores to compute it. Therefore, an emerging branch of research has decided to remove the softmax entirely.

Without softmax, our attention equation becomes 

$$ o = \sum_{i=1}^t (q_tk_i^T)v_i = q_t \sum_{i=1}^t k_i^Tv_i = q_t S_t$$

where $S_t$ is some state matrix at timestep $t$. We note that 

$$S_t=\sum_{i=1}^t k_i^Tv_i=\sum_{i=1}^{t-1} k_i^Tv_i + k_t^Tv_t = S_{t-1} + k_t^Tv_t$$

so we can compute $S_t$ at each timestep by repeatedly adding a rank-1 outer product matrix. 

A bonus of using linear attention is that at inference time, softmax attention needs to compute the kv cache, which takes $O(Td)$ memory. However, we can just keep updating the state matrix for $O(d^2)$ memory, which is usually better.

Implementation is easy. Yippee!

```
kv = torch.einsum('bti,btj->btij', k, v)
kv = torch.cumsum(kv, dim=1)
out = torch.einsum('bti,btij->btj', q, kv)
```

## Actually optimizing for hardware

We've removed softmax, but in terms of using a more hardware-efficient algorithm, we haven't made that much progress. Due to the recurrent nature of the $S_t$ calculation, we can't parallelize this task across multiple GPUs well. Fortunately, our current algorithm is in a much better position to be paralellized: we can split the sequence into chunks. For sake of example, suppose each chunk size was $c$ and we have 2 GPUs. Then GPU 0 can aggregate the state matrices from timesteps $1$ to $c$, and GPU 1 can aggregate the state matrices from $c+1$ to $2c$. Now, to calculate $S_{2c}$, GPU 1 just needs to aggregate its results with the results of GPU 0, leading to only $O(c)$ matrix multiplications done per GPU instead of $O(T)$ on one. 

## Improving performance

We've helped making the training much more efficient, but removing the nonlinearity costs us performance. In particular, we don't use any positional embedding since the signal from the position would get mixed up in the state matrix.

### Option 1: Decay

We generally want to have some recency bias to focus on tokens that are closer to our current one, so we could add some scalar $\gamma$ gate to scale the state matrix by each step. This would work, and we instead of a scalar gate we can use a learned matrix. In particular, we learn a vector that we use to outer product with the embeddings.

### Option 2: Better Updates

We need to be able to associate the key and value vectors, since the key is responsible for telling us how much of the value vector the current token uses. In softmax attention, we used a KV cache to store the vectors, but one of linear attention's main selling points is that we don't need the cache. 

Instead, maybe we can learn the association between the key and value vectors. Let $f_{\theta}(k)=kW$ for some weight matrix $W$, and let's use for now a cosine similarity-like loss without normalization for now. We know $L_t(W)=-(k_tW)^Tv_t$, and using a standard SGD update we get $W_t=W_{t-1}-\alpha \nabla L_t(W_{t-1})=W_{t-1}+\alpha k_t^Tv_t$. 

Wait isn't this exactly how we update the state matrix? This means that in fact, our state matrix is our regression function from key to values. 

## Delta Update Rule

This observation is really cool, but there is a huge flaw: the loss doesn't have normalization, so we can lower the loss by choosing arbitrarily large predicted value vectors. Instead, let's consider a much more sensible loss that is the squared L2 norm of the difference of the vectors ($\frac{1}{2}||k_tS-v_t||^2$).

Rewriting the update again, we get

$$S_t=S_{t-1}-\beta \nabla L(S_{t-1})=S_{t-1}+\beta k_t^T(v_t-k_tS_{t-1})$$

We call this the Delta Update Rule.

Then our algorithm becomes clear. First, we compute $v_{old}=k^T_tS_{t-1}$, the key we predict before updates. We learn our parameter $\beta$ through a matmult + sigmoid from our embeddings and calculate $v_{new}=(1-\beta) v_{old}+\beta v_t$. Finally, we remove the contribution of $v_{old}$ and replace it with the contribution of $v_{new}$ (the contribution being the outer product with $k_t$).

## Chunkwise Parallel Form Revisited

We've seen the advantages of the chunkwise parallel form already, so let's try to reformulate our Delta Update Rule in a way that is conducive to hardware-efficient training.

$$S_{t-1}+\beta(v_t-k_tS_{t-1})=(I-\beta k_t^Tk_)S_{t-1} + \beta k_t^T v_t$$

Now, we can do some simplification. Call $A_t=I-\beta k_t^Tk_t$ and $B_t=\beta k_t^Tv_t$, then our update is of the form $S_{t}=A_tS_{t-1}+B_t$. This draws out the recurrence relation, so we have some more idea of how we can aggregate results across GPUs. Unrolling the recurrence, we get 

$$S_t=(A_tA_{t-1} \cdots A_1)B_1+(A_tA_{t-1} \cdots A_2)B_2 + \cdots + B_t$$

Compressing this form, we get 

$$S_t=\sum_{i=1}^t (\prod_{j=i+1}^t A_j) B_i$$

Now, we isolate the operations done per chunk

$$S_t=(\prod_{j=C+1}^t A_j) \sum_{i=1}^C (\prod_{j=1}^C A_j) B_i + \sum_{i=C+1}^t (\prod_{j=i+1}^t A_j) B_i$$

which we know is equal to 

$$S_t=(\prod_{j=C+1}^t A_j) S_C + \sum_{i=C+1}^t (\prod_{j=i+1}^t A_j) B_i$$

In general, $\prod_{j=C+1}^t A_j$ is still difficult to compute. But we know $A_j=I-\beta k_j^Tk_j$, which is the difference between a diagonal matrix and a rank 1 matrix. We can therefore rewrite this as $I - \sum_{j=C+1}^t k_j^T w_j$ for some $w_j$. With some optimizations, this can be done quickly.

## Test-Time Training

We have reformulated the state matrix update as learning a linear regression function. But right now, everything is very limited: our model is purely linear, we have a fixed loss, and we process just one update per step. ML has come so far from the early days, surely we can use some of those advancements now?

This has led to an interesting field of research known as Test-Time Training (TTT). Right now, it is very much open, and there isn't a single best agreed way to do TTT yet. I have added the implementation for the RWKV series models, which to my knowledge is one of the most extensive lines of linear attention models. Recently, there have been ideas to merge softmax and linear attention, with Qwen3.5 being a major "hybrid attention" model. Maybe I will get to that someday, but for now I think I have paid enough attention to attention.

## Remarks

It's a little bit funny to me how an innocent idea to improve attention could lead to neural networks inside neural networks, but I guess that's what researching ML does to you. I think TTT is especially interesting because it leaves the door open to personalized LLMs. Like if we could run Qwen 3.5 9B on our computers, and it could learn our personalities and preference and what not, then what would the field of LLM memory research be? 