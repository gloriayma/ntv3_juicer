i have a bone to pick with the ntv3 track modelling head. my understanding of it is you basically learn a linear projection per track. but there's no inducitve bias telling different RNAseq projections to be more similar to each other than eg the ATAC projections. and also, there's no inductive bias telling the model that "this RNAseq track comes from thes ame tissue as this ATAC track".


i would like to do this experiment: look at the learned projections of the ntv3 heads, and try to cluster. does it cluster based on modality, or does it cluster based on learned cell type etc



Okay, so in my head, this is the plan. I'm going to load in the model and just get the tensors for the head and specifically only the rows that correspond to human because that is what I am interested in right now. And then I'm going to download the metadata thing to see what sort of tracks with what modalities and what cell types we have. And then I'm going to look through the metadata and see between experiments of different modalities, are the cell types, can they be matched to each other? So clearly we can slice the human tracks along the modality axis, but does it still work if we slice along the cell type or tissue type axis? Next, I'm going to do a naive UMAP on the embeddings, or sorry, not the embeddings, but the linear learned things per track. First, I'm going to normalize to mean zero and standard deviation one because you said so, but actually I'm not sure why we should be doing that. It almost seems to me that we shouldn't be doing that. And then we do the UMAP and we color by modality and we expect to see that at least the different modalities should cluster. And then we investigate what PCA1 is and whether it's the modality. And then we remove PCA1 and see what things cluster by. And we want a similarity grid between these linear learned things per track. So the way we're going to order our X and Y in this grid is going to be first by modality and then by cell type or tissue location, and with randomly permuted modality and see if a grid still emerges. And then the last thing I understand is this conditioned on these two things being from these two linear embeddings or linear projections being from the same cell type, is their similarity going to be more similar than if they are from different cell types? So what this will look like is, okay, so we have a big pairwise grid of cosine similarities between all of the different tracks. And now the question is, let's condition on these cells being from the left lobe of the liver. Then let's compute the average pairwise cosine similarity between all tracks that come from left lobe of the liver, and then we compute the average cosine similarity between these left liver tracks and all other tracks. So say we have five left liver tracks and 95 other tracks. So first we compute the expected cosine similarity within the five liver track thing. So that's going to give us an average of 25 cosine similarities. And then we compute the average cosine similarity between a liver track and a non-liver track. So that's going to give us five times 95 cosine similarities, and then we average that. And then we compare these two numbers, and we do this over all biological tissues, and we see whether the deltas are positive. These seem to be the three things that I want to run.

Yes. Those are the right three broad analyses, but I would change a few details so the result answers the question you actually care about.

First, I would not z-score each head vector to mean 0 and SD 1 by default. If \(w_t\in\mathbb R^d\) is the learned vector for track \(t\), subtracting its own mean and dividing by its own SD changes its direction, which is exactly the thing you want to study. For cosine-based analyses, the natural preprocessing is simply

$$ \tilde w_t=\frac{w_t}{\|w_t\|_2}. $$

Then cosine similarity is just \(\tilde w_i^\top\tilde w_j\). I would probably inspect both raw norms and normalized directions, because the norm of \(w_t\) may itself carry information.

For the UMAP, use the head vectors with cosine distance and color the same embedding separately by modality and by tissue/cell type. That's a useful exploratory figure. PCA is also useful, but I would not assume that "PC1 = modality" or remove PC1 just because modality appears correlated with it. Modality is categorical and may occupy several dimensions. A better quantitative question is: how predictable is modality from the PCs, and how predictable is tissue/cell type? If later you want to explicitly remove modality, regress modality out of the entire weight matrix or remove a learned modality-predictive subspace rather than deleting one PC.

Your similarity-grid idea also makes sense. Compute

$$ S_{ij}=\cos(w_i,w_j). $$

Then visualize the identical matrix under different row/column orderings:

modality → tissue
tissue → modality

That alone can be quite revealing. Randomly permuting the labels/order is useful as a visual sanity check, but it doesn't constitute a statistical null: the similarity matrix itself hasn't changed. The actual permutation test should randomize tissue labels, modality labels, etc., and recompute whatever summary statistic you're testing.

The main correction is to your third analysis. Consider your five left-liver tracks. Within those five, you don't want \(5\times5=25\) similarities because that includes each track compared with itself and double-counts \(i,j\) and \(j,i\). There are

$$ {5\choose2}=10 $$

unique non-self pairs.

More importantly, the comparison

$$ E[S_{ij}\mid\text{same liver}] - E[S_{ij}\mid\text{one liver, one non-liver}] $$

doesn't yet answer your cross-modality question. Imagine the five liver tracks contain three RNA-seq replicates. Those RNA-RNA similarities could make the within-liver number high simply because RNA heads cluster with RNA heads.

You want something closer to

$$ \Delta_c = E[S_{ij}\mid c_i=c_j=c,\;m_i\neq m_j] - E[S_{ij}\mid c_i=c,\;c_j\neq c,\;m_i\neq m_j], $$

where \(c\) is tissue/cell type and \(m\) is modality.

Even that should ideally be matched on modality pair. For example, if liver has RNA+ATAC, compare:

$$ \underbrace{\cos(w_{\text{liver,RNA}},w_{\text{liver,ATAC}})} _{\text{same tissue}} $$

against

$$ \underbrace{ E_{c\neq\text{liver}} [\cos(w_{\text{liver,RNA}},w_{c,\text{ATAC}})] }_{\text{different tissue, same RNA--ATAC comparison}}. $$

That gives you

$$ \Delta_{\text{liver}}^{\text{RNA,ATAC}}. $$

Repeat for RNA–DNase, RNA–ChIP, ATAC–ChIP, etc. Then aggregate across modality pairs and tissues. This prevents differences such as "RNA–ATAC is intrinsically more similar than RNA–ChIP" from masquerading as tissue structure.

So I would formulate your three experiments as:

Geometry exploration: PCA/UMAP of \(L_2\)-normalized head vectors; color by modality and tissue/cell type.
Similarity structure: cosine-similarity matrix, shown under modality-first and tissue-first orderings.
The actual hypothesis test: holding the pair of modalities fixed, ask whether same-tissue pairs are more similar than different-tissue pairs.

The third is the strongest one. The first two tell you what the geometry looks like; the third directly tests the claim that NTv3 has spontaneously learned a shared biological-context representation despite not being given an explicit tissue × modality factorization.
