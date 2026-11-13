# SEBot (Yang et al., KDD 2024): implementation spec

Label key: **[P-paper]** read in the paper (arXiv 2405.11225v1; the KDD PDF is not freely downloadable
and is assumed identical), **[P-code]** read in `846468230/SEBot` at commit
`59b5407d2d5600aee06f507c2b0d343fa599b482` (paths relative to the repo root), **[S]** secondary,
**[U]** unverified or assumed.

## 1. Summary

SEBot is a transductive, full-batch, graph-based detector on the BotRGCN feature scheme. It has three
views, each with its **own** copy of the BotRGCN-style feature encoder:

- **α view (whole-graph encoding tree, "SEP-U").** A fixed-height encoding tree of the whole user graph
  is built by structural-entropy minimisation. A GCN U-Net then pools nodes up the tree and unpools
  them back down.
- **β view (subgraph encoding trees, "SEP-G").** Each labelled user gets a small ego subgraph with its
  own encoding tree. A GCN with tree pooling and readout turns it into a subgraph embedding.
- **γ view (heterophily-aware relational encoder on an edge-dropped graph).** This is the R-GCN-like
  "FACNConv" with signed (tanh/Gumbel) edge weights and relational channel-wise mixing (RCM).

The training loss is cross-entropy on [h_α ‖ h_β ‖ h_γ] plus two InfoNCE terms: α vs γ and β vs γ.
Encoders: frozen RoBERTa-base features from the botrgcn scheme. There is no temporal component. The
expensive and novel part is the encoding trees. They are a label-free preprocessing step that we must
rebuild ourselves, because the authors' trees are only on SharePoint.

## 2. Citation and code

- Yingguang Yang, Qi Wu, Buyun He, Hao Peng, Renyu Yang, Zhifeng Hao, Yong Liao. "SEBot: Structural
  Entropy Guided Multi-View Contrastive Learning for Social Bot Detection." KDD 2024, pp. 3841-3852.
  DOI 10.1145/3637528.3671871, arXiv 2405.11225.
- Code: https://github.com/846468230/SEBot, commit `59b5407d…` (last push 2025-02-20). The licence is
  **Apache-2.0** (verified: `LICENSE` file and GitHub API). We may reuse it with attribution, a copy of
  the licence and a statement of changes. There is no NOTICE file. Recommendation: port
  `SEPG/utils.py::PartitionTree` (the dense coding-tree code) with an attribution header, both to build
  the tiny subgraph trees and as the oracle for the sparse whole-graph builder. Write everything else
  ourselves.
- The README links three SharePoint folders (dataset, trees, subgraphs). They were not tested **[U]**.
  We do not depend on them.
- Pinned versions: torch 1.13.0, torch_geometric 2.2.0, torch_sparse 0.6.16, numba 0.56.4, networkx 3.0
  (`requirements.txt`) [P-code].
- Datasets in code: `main_twibot20.py` (TwiBot-20) and `main_mgtab.py` (MGTAB) only. `SEPN/tree.py`
  and `SEPG/utils.py` handle `'twibot-20'` or else MGTAB `features.pt`. There is **no TwiBot-22 code,
  and the paper reports no TwiBot-22 numbers** [P-code][P-paper]. This confirms the survey.
- Algorithmic lineage: structural entropy and encoding trees from Li & Pan 2016. The coding-tree code
  (`build_coding_tree(mode='v2')`, leaf-up/root-down growth) is the SEP code of Wu et al. 2022 (ICML,
  arXiv 2206.13510), which follows the HCSE stretch/compress scheme [P-code; lineage S].

## 3. Inputs

### 3.1 Node features: feature scheme `botrgcn` (see botrgcn.md)

`main_twibot20.py` l.289-295 builds `x = cat([num_properties(5), tweets(768), cat_properties(3),
des(768)])`, 1,544 columns, in that order. `args.prop_num=5`, `cat_num=3`, `des_num=tweet_num=768`
(l.386-389) [P-code].

- **Deviation to note.** The widths 5 numeric and 3 categorical match the TwiBot-22 benchmark BotRGCN
  scheme (followers, active_days, screen_name_length, following, statuses; protected, verified,
  default_profile_image). They do *not* match the original BunsenFeng/BotRGCN TwiBot-20 tensors
  (6 numeric, 11 categorical). The benchmark `src/BotRGCN/twibot_20/preprocess_1.py` also has a
  copy-paste bug that would give one categorical column reshaped to 5,301 rows. So the SharePoint
  TwiBot-20 tensors were built by some third script **[U]**. Use the `botrgcn` scheme with 5 numeric
  and 3 categorical columns for TwiBot-20, as botrgcn.md defines it. Make `prop_num` and `cat_num`
  follow that scheme's widths.
- **Row order (required).** Labelled users come first, in the order train (8,278), val (2,365),
  test (1,183), then support (217,754). The code hard-codes the split as
  `train=first int(0.7·11826)=8278`, `val=next 2365`, `test=last 1183` (l.296-304). The shared layer's
  `users.parquet` order (labelled first) satisfies this if, within the labelled block, rows are sorted
  train, val, test. Otherwise pass explicit index arrays. The 70/20/10 slicing only reproduces the
  official sizes.
- `label.pt` covers only the 11,826 labelled rows (`node_num=11826`).

### 3.2 Graph

- Relations: TwiBot-20 `following` and `follower`, exactly as in the botrgcn `edge_index`/`edge_type`
  (2 relations, 227,979 directed edges per paper Table 1) [P-paper]. From the shared layer:
  `graph/following.csr.npz` and `graph/follower.csr.npz` (TwiBot-20).
- **γ view and SEP-U level-0 graph.** `relational_undirected` (l.41-57): for each relation it takes
  the unique directed edges, adds their reverses, and concatenates the relations. Edge types are kept,
  and a pair present in both relations appears in both. This is the `data.edge_index` used by FACNConv
  (per relation) and by SEP-U's first and last GCNConv (types ignored, so cross-relation duplicates
  count twice in GCN normalisation) [P-code].
- **Encoding-tree graph.** `SEPN/tree.py::load_tree` (l.126-141) builds `to_undirected(edge_index)`
  over both relations into an `nx.Graph`. The result is simple, unweighted (duplicates collapse to
  weight 1), has self-loops removed (l.15-19), and has N = 229,580 nodes including isolated ones
  [P-code].
- **Subgraphs.** These use the raw *directed* `edge_index` with both relations merged. See §3.4.

### 3.3 Whole-graph encoding tree (α view): what to build

The model consumes, for a tree of height k (`tree_depth`), the output of `SEPN/tree.py::extract_layer_data`:

- `node_size[d]`: number of tree nodes at level d. Level 0 holds the leaves, i.e. the N graph nodes in
  index order; level k is the root.
- `interLayer_edgeMat[d]` (d ≥ 1): LongTensor [2, node_size[d-1]] of pairs (parent local index at
  level d, child local index at level d-1). Every node at level d-1 has exactly one parent.
- `layer_edgeMat[d]` (d < k): the edge list of the quotient graph at level d (`nx.quotient_graph` of
  level d-1 by the level-d partition), symmetrised, unweighted, without self-loops. Level 0 is the
  graph itself.

**The tree must be balanced**: every leaf is at depth exactly k, with unary chain nodes where needed.
The code achieves this through `check_balance`/`single_up` and sets `depth := child_h`.

**Which algorithm the code runs** [P-code `SEPN/codingTree.py`]:
`PartitionTree(adj).build_coding_tree(k)` with the default `mode='v2'` (l.493). The paper's
Algorithm 1 (binary merge to full height, then "Drop"/compress to height k) is the code's `mode='v1'`.
**Paper and code differ. Implement v2 (code default); v1 is an optional ablation.**

Definitions. vol(X) = sum of degrees in X; g(X) = cut(X, V∖X); VOL = vol(V). For a tree node α with
parent α⁻: H^T(G) = −Σ_α (g_α/VOL) log2(vol_α/vol_α⁻) [P-paper eq.3]. The whole-graph code
(`SEPN/codingTree.py`) adds **+1 smoothing** so that isolated nodes (vol 0) do not produce log 0:

- `CombineDelta(a,b,cut)` = [ (v1−g1)·log2(v12/v1) + (v2−g2)·log2(v12/v2) − 2·cut·log2(Gvol/v12) ] / Gvol,
  with v=vol+1, g=g+1, v12=v1+v2 (l.116-126).
- `CompressDelta(n, p)` = child_cut(n) · log2((vol_p+1)/(vol_n+1)) (l.109-113).
- `entropy()` and `leaf_up`/`root_down` use (vol+1) ratios (l.230-241, 189-190, 214).

The subgraph copy (`SEPG/utils.py`) is identical except that it has **no +1** anywhere [P-code;
verified by a whitespace- and comment-insensitive diff]. Keep both variants, selected by a flag.

Algorithm (v2), as the code runs it:

1. **`__build_k_tree(Gvol, nodes, k=2)`** (l.243-369).
   - *Merge phase.* Push (CombineDelta, i, j, cut) for every adjacent pair i<j. Repeatedly pop the
     minimum; skip the pair if either side is already merged. Otherwise create a new node with a fresh
     sequential id, children {i,j}, vol=v_i+v_j, g=g_i+g_j−2·cut, child_h=max+1, child_cut=cut. Its
     adjacency is the union of the children's. Push CombineDelta for every unmerged neighbour (cut
     recomputed). For each child with child_h>0, push (CompressDelta(child,new), child, new) to the
     compress heap.
   - When the heap is empty but more than one community is unmerged (disconnected graph or isolated
     nodes), add a root whose children are all unmerged nodes, with vol=Gvol and g=0, and push their
     compress entries.
   - *Compress phase.* While child_h(root) > k: pop the minimum CompressDelta. If that node's
     depth+height ≤ k, skip it. Otherwise splice it out (its children move to its parent, and the
     parent's child_cut gains its child_cut), then update heights. If child_h(root)==k, stop. Otherwise
     refresh the affected heap entries and re-heapify (l.346-368).
   - Ties are broken by tuple order (diff, id1, id2, cut), so the result is deterministic.
2. `check_balance(root)`: every root child that is a leaf gets a unary parent (`single_up`).
   Then set child_h(root) ≥ 2.
3. While child_h(root) < k, compare two candidate expansions (l.506-540):
   - **leaf_up.** For every height-1 node with at least 3 leaves, run `__build_k_tree(k=2)` on the
     induced subgraph of its leaves and score the entropy decrease. Total Δ is divided by N. Nodes
     with 1 or 2 leaves get unary parents.
   - **root_down.** Run `__build_k_tree(k=2)` on the graph whose nodes are the root's children
     (community adjacency, cuts between leaf sets). Δ is divided by the number of root children.
   - If leaf_up Δ < root_down Δ, apply root_down; otherwise apply leaf_up. The root height grows by 1.
     Only the expansion just applied is recomputed in the next round (flag logic).
4. Relabel nodes level by level (`update_node`), sorted by (depth, old id).

**Build it sparse.** This is mandatory. The official code cannot run at our scale:
`nx.to_numpy_array` gives a dense 229,580² float64 matrix of 421.6 GB, and `graph_parse` is a
5.3·10¹⁰-iteration Python double loop. `cut_volume` is O(|P1|·|P2|) over leaf lists, and each merge
copies the `partition` list, which is O(N·tree-height). The authors had 264 GB of RAM, so the shipped
TwiBot-20 tree cannot have come from this script as written **[U]**.

Sparse design:

- Input is an undirected CSR (int32 indices) with degrees.
- Each active community keeps a hash map neighbour-community → cut weight. On merge, fold the smaller
  map into the larger; cut(A∪B, C) = cut(A,C) + cut(B,C). This replaces `cut_volume`.
- Do not store leaf lists. Keep `parent`, `children`, `vol`, `g`, `child_h`, `child_cut` arrays of
  size at most 2N plus the unary nodes.
- Use a lazy-deletion binary heap with periodic compaction when stale entries exceed 2× the live
  entries.
- leaf_up runs on the induced leaf subgraph of each height-1 community. Build it from the CSR. Run the
  communities independently, possibly in parallel.
- root_down runs on the community graph of the root's children, built from the level-1 adjacency maps.
- Write it in numba (or C++/pybind). Pure Python is acceptable only for TwiBot-20.

**Equivalence test.** On random and real subgraphs (n ≤ 3,000, including isolated nodes and
disconnected parts), the sparse builder must produce exactly the same tree as the ported dense
`PartitionTree` for k = 2…8, in both the +1 and no-+1 variants. The heap keys are identical, so this
is achievable. Remaining differences can only come from floating-point summation order; use float64
and the same expression order.

**Hub caveat.** The merge phase pushes deg(new community) heap entries per merge, so a community
adjacent to a hub costs O(deg) per merge. That is fine on TwiBot-20 (max degree is in the thousands
[U]). On TwiBot-22 it may be quadratic: see §7.

### 3.4 Subgraphs and subgraph trees (β view)

Source: `SEPG/utils.py::load_graph` (l.582-655), `SEPG/trans_graph.py`, `SEPG/data.py` [P-code].

1. Take the directed `edge_index` with both relations merged (types ignored).
   `limit_neighbor_num(edge_index, top_n=10)` keeps, for each **source** node with more than 10
   out-edges, a random sample of 10. The draw uses `np.random.choice` with no seed set (l.551-569).
   Then remove self-loops.
2. For each labelled node i < `node_num` (11,826): `k_hop_subgraph(i, num_hops=hop,
   edge_index, relabel_nodes=True)` with PyG's default `flow='source_to_target'`. It collects nodes j
   with an edge **j → i** (in-neighbours) and the induced edges among them. **`hop = 1` is hard-coded**
   (l.594). The paper's Table 2 says subgraph order m = 2 for TwiBot-20 and m = 1 for MGTAB.
   **Recommend m = 1 (code)**, with m = 2 as an ablation.
3. **Lone node** (no edges in its subgraph): build a star of i plus `top_n = 10` random *labelled*
   nodes (uniform from 0…node_num−1, distinct, ≠ i, unseeded `random.randint`) with edges (0, h+1)
   (l.618-632). Then `to_undirected`.
4. Make an `nx.Graph`. Build its encoding tree with `PartitionTree` (no-+1 variant), `mode v2`,
   height k (same k as the whole graph). Then `extract_tree`: `node_size`, `edges[d]` (parent, child)
   pairs, `graph_mats[d]` (quotient-graph edges for d < k), `nodelist` (global ids, local order).
   The tag/degree one-hot features in `data.py` are computed but **unused by the model**, because
   SEP-G gathers `x[nodelist]`.

Important behaviour [P-code + inference]. In botrgcn TwiBot-20 tensors every edge originates at a
*labelled* user (the neighbour lists belong to labelled users) **[U, check on our edge layer]**. So the
in-neighbours of a labelled node are other labelled users who list it. Most labelled nodes are then
probably lone nodes that get a **random 10-node star**, and the β view is close to noise for them.
Reproduce this behaviour in faithful mode, and log the lone-node count. Ablation: undirected 1-hop
neighbourhood capped at 10 random neighbours.

### 3.5 What we need from the shared data layer

`users.parquet` (idx, split, label; row order as in §3.1). The botrgcn tensors (see botrgcn.md):
`emb/roberta-base/des.f16.npy`, `emb/roberta-base/tweets_mean.f16.npy`, plus the 5 numeric and 3
categorical properties. `graph/following.csr.npz` and `graph/follower.csr.npz`. `extscore/<tag>.parquet`.
No per-tweet text, no timestamps.

Extra artifacts built by this method, under `artifacts/methods/sebot/<dataset>/`:

- `tree_k{k}_{v2|v1}_{smooth}.npz`: per level, `parent` int32 arrays and quotient-graph CSRs.
- `subgraphs_hop{m}_top10_seed{s}.npz`: ragged `nodelist`, per-level `node_size`, parent arrays,
  quotient-edge arrays, with offsets.

Neighbour capping and lone-node fill take an explicit seed (the authors' run was unseeded).

## 4. Architecture (TwiBot-20 defaults; d = hidden_dim = 32)

**FeatureEncoder** (three independent copies: in `SEP_U`, `SEP_G` and the γ backbone) [P-code
`SEPN/sep_u.py` l.15-61, `main_twibot20.py` l.75-89, 111-123]:

- des 768→32, tweet 768→32, num 5→32, cat 3→32, each Linear followed by LeakyReLU (input_dim=128,
  so 128/4 = 32 each).
- Concatenate in the order (des, tweet, num, cat) → Linear 128→32 → PReLU(32 channels).

**γ view, the "BotRGCN" backbone with FACNConv** [P-code `main_twibot20.py` l.60-129,
`BackBone/rgcn.py` l.89-177]:

- In training mode only: `edge_mask` drops each edge independently with probability `pe = 0.2`, once
  per forward pass, shared by both layers.
- FeatureEncoder → FACNConv(32,32,R=2) → dropout 0.5 → FACNConv(32,32) → dropout 0.5 →
  Linear(32,32). **There is no activation between the conv layers.**
- **FACNConv(x)**:
  - root = Linear(32,32, bias)(x).
  - For each relation r: x_r = x·W_r (W_r is [32,32] with no bias). Messages run along
    edge_index[0] → edge_index[1] on relation-r edges.
  - Attention a = [x_r,i ‖ x_r,j] · att_r with att_r of shape [64,1]. It uses the *transformed*
    features; paper eq.11 uses h^{l−1}.
  - Add logit(ε) with **ε = torch.rand(1): one scalar per relation per call, also drawn in eval mode**.
  - w = tanh((a + logit ε)/τ_G) with τ_G = 0.01 (constructor default; the CLI does not set it).
  - Apply dropout(p=0.3, training) to w.
  - message = x_r,j · w / indeg_r(i), with indeg clamped ≥ 1; aggregate by sum.
  - **RCM:** U = LeakyReLU(Linear(2·32 → 2·32)([h_1‖h_2])), reshape to [N,2,32], softmax over the
    relation axis. Output = Σ_r h_r ⊙ U_r + root.
- Initialisation: kaiming_uniform for root, W, the RCM linear and att; zero biases.

**α view, SEP_U** (num_blocks = 2) [P-code `SEPN/sep_u.py` l.94-218]:

- x0 = FeatureEncoder(x), [N,32].
- Down, for b = 0,1:
  - x = Dropout0.5(ReLU(GCNConv_b(x, E_b))), where E_0 = `data.edge_index` and E_1 = level-1 quotient
    edges. Save skip_b = x.
  - x = Dropout0.5(ReLU(SEPool_b(x))). SEPool sums children into parents over interLayer[b+1], then
    applies the MLP Linear(32,32) → ReLU → BatchNorm1d(32).
- Up, for level l = 2,1:
  - x = Dropout0.5(ReLU(GCNConv(x, E_l))), with input 32 at l=2 and 64 at l=1.
  - x = Dropout0.5(ReLU(SEUnpool(x))). The unpool copies the parent vector to each child, then
    applies a separate MLP of the same form.
  - x = cat[x, skip_{l−1}] (64).
- Final: GCNConv(64→32, E_0), then Linear(32,32). Output h_α is [N,32]; slice to the first node_num
  rows.
- GCNConv is PyG's default (self-loops, symmetric normalisation, bias).
- Only tree levels 0-2 are used, whatever k is. k affects h_α only through the bottom two partitions.

**β view, SEP_G** (num_convs = 3, conv='GCN', global_pooling='average') [P-code `SEPG/sep_g.py`
l.101-253]:

- x = FeatureEncoder(x_all). Gather x[nodelist] for all 11,826 subgraphs, batched by offsets.
- For c = 0,1,2:
  - x = ReLU(GCNConv_c(x, graph_mats[c])), 32→32.
  - If c < k−1: x = ReLU(SEPool(x)). This is a pure sum of children into parents, **no MLP** (the MLP
    is commented out, l.85-88).
  - Save xs[c].
- Readout: mean-pool xs[c] per subgraph at level min(c+1, k−1). Concatenate to 96 →
  Linear(96,32) → ReLU → Dropout(0.5) → Linear(32,32). Output h_β is [11826,32].
- This view has no dropout other than the classifier's. It needs k ≥ 3.

**Heads** [P-code `main_twibot20.py` l.132-236]:

- Classifier: Linear(96,32) → LeakyReLU → Linear(32,2) on [h_α ‖ h_β ‖ h_γ], in that order.
- proj_u and proj_g: Linear(32,16) → LeakyReLU → Linear(16,32). Each head is applied to **both**
  views of its pair.
- L_NCL = InfoNCE(proj_u(h_α), proj_u(h_γ)); L_SCL = InfoNCE(proj_g(h_β), proj_g(h_γ)). Both are over
  all node_num labelled rows, including val and test (label-free and transductive).
- InfoNCE is NT-Xent: L2-normalise, cosine similarity over the 2N×2N matrix. The positive pair is
  (i, i+N). The denominator sums exp(s/τ) over all k ≠ i, which includes the positive. Average over
  the 2N rows. τ = 0.1.
- Loss = CE(train rows, unweighted, 2-class softmax) + λ1·L_NCL + λ2·L_SCL.

## 5. Training

| Item | Paper Table 2 (T-20) | Code default (`main_twibot20.py` l.380-425) | Use |
|---|---|---|---|
| optimiser | AdamW | AdamW | AdamW |
| lr | 0.01 | 0.01 | 0.01 |
| weight decay | 3e-3 | 3e-3 | 3e-3 |
| dropout | 0.5 | 0.5 (also conv/pooling dropout 0.5) | 0.5 |
| hidden | 32 | 32 (input_dim 128, proj 16) | 32 |
| λ1, λ2 | 0.09, 0.03 | 0.1, 0.1 | **paper** |
| tree depth k | 6 | 4 | **paper (6)**; sweep 3-8 |
| subgraph order m | 2 | 1 (hard-coded in `SEPG/utils.py`) | **code (1)** |
| epochs | 70 | 80 (patience 50 on val loss) | **paper (70)** |
| τ / τ_G | 0.1 / 0.01 | 0.1 / 0.01 | same |
| edge drop pe | n/a | 0.2 | 0.2 |
| LR schedule | n/a | CosineAnnealingLR(T_max=16, eta_min=0), stepped per epoch; periodic past 16 epochs | code |
| seed | n/a | 42 | 5 seeds: 42, 43, 44, 45, 46 |

The CLI values used for the reported runs are unknown. The code defaults visibly differ from Table 2,
so the defaults were not the reported configuration. We therefore take the paper's values for every
hyperparameter that Table 2 lists and the code for everything else. MGTAB settings (Table 2): λ 0.05
and 0.05, m = 1, k = 6, 200 epochs. `main_mgtab.py` instead shuffles a random 7:2:1 split with the
seed, and uses pe = 0.3 and dropout 0.3.

- **One optimiser step per epoch, full batch.** Evaluate every epoch on val and test.
- **Model selection:** report test metrics at the epoch with the **minimum validation loss**. The code
  also prints the 6 lowest-val-loss epochs (l.362-373). Patience 50 never triggers within 70 epochs.
- Determinism: `cudnn.deterministic` is on, but scatter-add is non-deterministic on GPU. Accept small
  run-to-run noise.

## 6. Evaluation and published numbers

Metrics are computed with sklearn on the test rows: accuracy, and binary F1/precision/recall with
bot = 1. **The F1 is bot-class F1.** The paper reports mean ± std, with the number of runs not stated
[P-paper Table 3].

| Dataset | Acc | F1 (bot) | Recall | Precision |
|---|---|---|---|---|
| TwiBot-20 (official split) | 87.24 ± 0.10 | 88.74 ± 0.13 | 92.97 ± 1.16 | 84.90 ± 0.79 |
| MGTAB (random 7:2:1) | 90.46 ± 1.44 | 82.12 ± 2.42 | 81.73 ± 2.77 | 82.52 ± 2.19 |

Ablations on TwiBot-20 (Acc/F1) [P-paper Table 4]:

| Variant | Acc | F1 |
|---|---|---|
| without the whole-graph tree | 86.39 | 87.84 |
| without subgraph trees | 86.45 | 88.02 |
| without RCM | 86.24 | 87.69 |
| R-GCN encoder | 86.35 | 87.96 |

The paper's BotRGCN re-run on TwiBot-20 gives 85.75 / 87.25. **No TwiBot-22 result exists, from the
authors or from the TwiBot-22 benchmark** (SEBot postdates it).

## 7. Scale plan (hard memory rule: every job runs through `tools/memrun`, MemoryMax 10G, no swap; design target ≤ 8 GB host RSS)

Text embeddings are inherited from the botrgcn scheme, so this method adds no encoder cost.

### TwiBot-20 (N = 229,580; about 228k directed edges; 11,826 labelled)

| Step | Host RSS peak | VRAM | Time |
|---|---|---|---|
| Load the CSR graph and build the undirected simple graph | indptr 1.8 MB + indices about 0.46M·4 B = 1.8 MB, so under 0.1 GB | n/a | seconds |
| Whole-graph tree, sparse numba, k = 6 (about 5 greedy passes) | adjacency maps about 0.46M entries·~40 B = 18 MB; heap ≤ 10M lazy entries·24 B = 240 MB; tree arrays 2N·40 B = 18 MB. Python-dict fallback is about 1 GB. Peak under 1.5 GB | n/a | numba: minutes. Python: under 1 h [U] |
| Subgraphs (11,826) and their dense trees with the ported `PartitionTree`, 14 worker processes | each worker about 150 MB of Python, plus dense adjacency s² float64 (s ≤ 2,000 gives 32 MB; cap in-degree at 2,000 and log if hit). 14 × 0.2 GB = 2.8 GB | n/a | about 11,826 × 20 ms / 14 ≈ 20 s to a few minutes |
| Training (full batch) | features fp16 memmap: 229,580 × 1,544 × 2 B = 0.71 GB; tree and subgraph arrays under 0.2 GB; torch with CUDA context about 2.5 GB. Peak about 4 GB | inputs fp16 0.71 GB (fp32: 1.42 GB); 3 feature encoders about 3 × 229,580 × (128+32) × 4 B × ~3 (saved for backward) = 1.3 GB; FACNConv edges ≈ 0.46M × ~200 B × 2 layers = 0.2 GB; InfoNCE: see below. Peak about 5 GB | 70 epochs × ~1-3 s; about 5 min per seed |

**InfoNCE must be chunked** (mathematically identical). With 2N = 23,652, one 2N×2N fp32 matrix is
2.24 GB. The official version keeps the similarity matrix, a float mask, the exponential and the
masked product for each of the two losses, about 13 GB in total, which does not fit in 12 GB.
Compute it by row blocks of 2,048: logits = z_blk·zᵀ/τ, self term set to −∞, then logsumexp. Wrap each
block in `torch.utils.checkpoint`. A block costs 2,048 × 23,652 × 4 B = 194 MB.

Cost: under 0.5 GPU-h for 5 seeds, plus at most 1 CPU-h of preprocessing.

### TwiBot-22 (N = 1,000,000, all labelled; following + followers about 3.74M directed edges [U, check on our layer])

This is an extension only. It is not reported anywhere, so there is no faithfulness target. Do it
only if needed.

- **Tree.** CSR: indptr 8 MB, plus about 6.6M undirected adjacency entries × 4 B = 27 MB. Adjacency
  maps about 6.6M × 40 B = 0.26 GB. Heap: cap at 50M lazy entries × 24 B = 1.2 GB and compact. Tree
  arrays about 80 MB. Host peak about 2-3 GB (numba).
  - **Pure Python is not allowed here**: tuple heap entries at about 90 B each would reach 4-5 GB.
  - Risk: hub communities make the merge phase about O(N · deg_hub). Run a pilot on a 100k-node
    induced subgraph and extrapolate. If it projects beyond 12 h per pass, switch to a
    version-stamped lazy re-evaluation (re-score a pair only when it is popped stale). Record that as
    a deviation.
- **Subgraphs.** 1M ego graphs. In-neighbourhoods of hubs are unbounded (100k+), and dense trees are
  impossible for them. **Cap in-neighbours at 10 random ones (seeded)**, as a deviation. Store them
  ragged on disk: about 1M × (≤ 21 nodes, ≤ ~60 edges, k levels) ≈ 1-2 GB, written as an `.npz`
  memmap in chunks of 50k egos. Workers: 14 × 0.2 GB.
- **Training cannot be full batch.**
  - Features fp16 on the GPU take 1M × 1,544 × 2 B = 3.1 GB.
  - FACNConv over about 7.5M symmetrised edges saves about 7.5M × (32+32+64+32) × 4 B ≈ 4.8 GB per
    layer.
  - InfoNCE over 1M rows is impossible.
  - Plan: mini-batch of 4,096 seed users per step. Use 2-hop neighbour sampling, [25, 10] per
    relation, for the γ view. Encode the α view per batch from tree levels 1-2: precompute the
    level-1 and level-2 community memberships and pool only the communities touched. InfoNCE runs
    within the batch (2 × 4,096). The β view runs per seed from its ego tree.
  - VRAM about 3.1 GB of features + about 2 GB of activations, so about 5-6 GB. Host RSS about 3 GB
    (memmaps, no pinned buffers beyond 2 × 4,096 rows, `num_workers` 0-2 at about 0.5 GB each).
- Estimate: CPU tree 2-12 h [U], subgraphs about 1 h, training about 3-6 GPU-h per seed [U].

## 8. ExtScore column

- Flag `extscore: none | p`. With `none`, the widths and computation are exactly as in §4.
- **Primary injection:** append [z, m] (shared-layer neural transform) to the numeric block, so
  prop_num goes from 5 to 7. Because `x` is shared, this feeds `linear_relu_num_prop` (7 → 32) in all
  three FeatureEncoders.
- Support nodes, and labelled nodes without a value, get z = 0 and m = 1. Fit μ and σ on observed
  TRAIN rows only.
- The column also influences InfoNCE (val and test rows): it is label-free, so there is no leakage.
- **Ablation A (late fusion):** concatenate [z, m] to [h_α ‖ h_β ‖ h_γ] (96 → 98) before the
  classifier. The α, β and γ encoders are unchanged.
- **Ablation B (γ only):** append [z, m] only in the γ backbone's encoder. The structural views stay
  feature-identical to the baseline.

## 9. Pitfalls

1. The tree builder must be sparse (§3.3). The shipped dense code is infeasible here (421.6 GB
   adjacency).
2. The algorithm runs `mode='v2'` (leaf-up/root-down). The paper's Algorithm 1 describes v1.
3. The whole-graph and subgraph tree code differ by the +1 volume smoothing.
4. Only tree levels ≤ 2 (SEP-U) and ≤ 3 (SEP-G) are used. Levels above that are computed but unused
   at k = 6.
5. The tree must be balanced (unary padding). `depth` is height above the leaves.
6. The subgraph rule follows in-edges only after capping *out*-edges at 10. Lone nodes get **random
   labelled-node stars**, unseeded in the original. The code uses hop = 1, the paper m = 2.
7. Three separate FeatureEncoders, not one shared.
8. FACNConv: a single scalar Gumbel/logistic noise per relation per call, **also at eval**. With
   τ_G = 0.01, tanh is close to a sign function with a random global threshold. So eval predictions
   are stochastic. Faithful mode keeps this; add a flag `eval_noise=false` for a deterministic
   ablation.
9. γ backbone: no nonlinearity between the two FACNConv layers. Edge drop applies only in training.
   Attention dropout is 0.3, which differs from the 0.5 feature dropout.
10. SEP_G pooling has no MLP; SEP_U pooling has an MLP with BatchNorm.
11. InfoNCE: the same projection head on both views; the denominator includes the positive. It must be
    chunked on 12 GB (§7).
12. LR schedule: cosine with T_max = 16 is **periodic**. The LR hits 0 at epochs 16, 48, … and peaks
    again at 32 and 64.
13. Model selection uses min val loss, but test metrics are computed every epoch. Report only the
    val-selected epoch.
14. The hard-coded split slicing needs the train, val, test row order. Labels exist only for
    11,826 rows.
15. The paper says h is 128-dimensional for visualisation (§5.5), while Table 2 and the code use 32.
    Use 32.
16. Transductive use: all 229,580 nodes' features and the full graph, including test nodes, enter the
    trees, the message passing and InfoNCE. This is label-free and standard for the benchmark, but
    note it.

## 10. Open questions / unverified

- The exact CLI arguments behind Table 3: tree_depth, hop, λ, epochs, number of seeds.
- The provenance of the SharePoint TwiBot-20 tensors: 5 numeric and 3 categorical columns, and their
  row order.
- How the authors' TwiBot-20 tree was computed at all, given the dense code.
- The edge direction convention in our edge layer versus botrgcn `edge_index`. It changes which
  in-neighbours form the subgraphs. Measure the lone-node count on TwiBot-20 (expected high) and the
  max degree.
- TwiBot-22 hub degrees and the merge-phase runtime (pilot needed).

## 11. Acceptance check

- **Primary, TwiBot-20 official split, `extscore: none`, 5 seeds, k = 6, faithful mode.** Mean accuracy
  in 87.24 ± 1.0 and mean bot-F1 in 88.74 ± 1.0. Also require: the full model is better than or equal
  to our R-GCN-encoder-only ablation minus 0.3 points, and better than or equal to our BotRGCN mean.
- **Secondary, MGTAB** (our feature tensors suffice; `main_mgtab.py` protocol: seeded random 7:2:1,
  `features.pt` 788-d, 7 relations): accuracy 90.46 ± 2.0 and F1 82.12 ± 3.0. No ExtScore is possible
  there.
- **Unit test:** the sparse tree builder equals the ported dense `PartitionTree` on graphs with
  n ≤ 3,000 (§3.3).
- TwiBot-22 has no target. Report it as an extension next to BotRGCN.
