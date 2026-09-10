# BA-Pred2 Project Specification

## 1. Goal

BA-Pred2 is a next-generation protein–ligand binding affinity predictor derived conceptually from BA-Pred, but redesigned around recurrent graph inference rather than a fixed stack of independent GNN layers.

The core hypothesis is:

> Binding affinity prediction benefits from repeatedly reconciling local molecular context and protein–ligand interface hypotheses with a shared recurrent graph operator.

The model should therefore support:

1. parameter-efficient recurrent depth,
2. test-time recycle scaling,
3. stable long-depth message passing without uncontrolled edge/node drift,
4. richer geometric graph structure than the original BA-Pred,
5. explicit separation of immutable molecular facts and dynamic latent interaction states,
6. optional E(3)-equivariant hidden-state updates without moving input coordinates,
7. reproducible preprocessing, training, evaluation, and ablation experiments.

BA-Pred2 is an affinity model, not a pose-refinement model. Coordinates are treated as fixed observations unless a separate future experiment explicitly enables coordinate refinement.

---

## 2. High-level architecture

The model is organized as

```text
Static graph construction
        ↓
Node/edge encoders
        ↓
Local chemistry prelude
        ↓
┌────────────────────────────────────┐
│ Shared recurrent interaction block │ × T
│                                    │
│ 1. Interface-state update          │
│ 2. Protein ↔ ligand cross message  │
│ 3. Protein intra propagation       │
│ 4. Ligand intra propagation        │
└────────────────────────────────────┘
        ↓
Ligand + interface + pocket readout
        ↓
pKd
```

The recurrent block uses the same parameters at every recycle step.

Let

- `H_P^t`: protein atom scalar hidden states,
- `H_L^t`: ligand atom scalar hidden states,
- `Q^t`: dynamic protein–ligand interface edge states,
- `V_P^t`, `V_L^t`: optional equivariant vector hidden states,
- `H_P^0`, `H_L^0`, `Q^0`: immutable encoded input anchors.

Then one recurrent step is conceptually

```text
Q^(t+1)              = F_interface(H_P^t, H_L^t, Q^t, H_P^0, H_L^0, Q^0, geometry)
Hbar_P^t, Hbar_L^t   = F_cross(H_P^t, H_L^t, Q^(t+1), H_P^0, H_L^0)
H_P^(t+1)            = F_protein_intra(Hbar_P^t, static protein graph, H_P^0)
H_L^(t+1)            = F_ligand_intra(Hbar_L^t, static ligand graph, H_L^0)
```

When the equivariant extension is enabled, the vector states are updated alongside scalar states while coordinates remain fixed.

---

## 3. Design principles

### 3.1 Static facts vs dynamic hypotheses

Static input features must not be overwritten by recurrent updates.

Static information includes:

- atom identity and chemistry,
- formal charge,
- aromatic/ring membership,
- covalent connectivity,
- spatial distances,
- positional/topological encodings,
- protein/ligand entity identity,
- original interaction flags,
- input coordinates.

Dynamic states include:

- contextualized atom representations,
- learned interface/contact representations,
- optional vector-valued directional context.

Every recurrent cycle receives the immutable encoded input again. This acts as an anchor against oversmoothing and recurrent state drift.

### 3.2 Recycle means refinement, not just depth

The recurrent operator should be interpretable as iterative binding inference:

```text
raw chemistry/geometry
→ first-order contact interpretation
→ binding-conditioned molecular context
→ contact cooperation/conflict
→ pocket–ligand consistency
```

The desired behavior is not merely to enlarge receptive field.

### 3.3 Bounded updates

Avoid unrestricted additive accumulation such as

```text
E^(t+1) = E^t + ReLU(F(E^t))
```

for dynamic interface edges.

Use gated interpolation or small LayerScale residuals instead.

### 3.4 No BatchNorm inside the recurrent core

Use LayerNorm or RMSNorm because the same shared operator sees different state distributions at different recycle depths.

---

## 4. Graph representation

Use PyTorch Geometric `HeteroData` with explicit protein and ligand node sets.

Recommended relation types:

```text
('protein', 'intra', 'protein')
('ligand',  'intra', 'ligand')
('protein', 'contact', 'ligand')
('ligand',  'contact', 'protein')
```

### 4.1 Protein intra graph

Construct

```text
E_P = E_P_covalent ∪ E_P_spatial
```

Covalent edges contain bond information when RDKit assigns a bond.

Spatial edges connect nearby non-bonded protein atoms within a configurable radius.

Recommended initial defaults:

```yaml
protein_spatial_cutoff: 5.0
protein_max_spatial_neighbors: 32
```

Each protein intra edge should distinguish

```text
covalent
spatial
both/overlap if represented explicitly
```

and include distance RBF features.

### 4.2 Ligand intra graph

Construct

```text
E_L = E_L_covalent ∪ E_L_spatial
```

Recommended defaults:

```yaml
ligand_spatial_cutoff: 4.5
ligand_max_spatial_neighbors: 24
```

For ligands, covalent topology remains especially important. Spatial edges should therefore carry an explicit edge-type encoding so that the model never confuses Euclidean proximity with a chemical bond.

### 4.3 Protein–ligand interface graph

Create a sparse bipartite contact graph for atom pairs within a cutoff.

Recommended default:

```yaml
interface_cutoff: 5.0
```

The raw interface graph is immutable. Only its latent state `Q^t` evolves.

---

## 5. Input features

### 5.1 Node features

Retain or improve on BA-Pred atom features. At minimum include:

- element / atom type,
- period/group or equivalent periodic information,
- degree,
- total hydrogens,
- hybridization,
- aromaticity,
- ring membership,
- formal charge,
- radical state if available,
- normalized atomic mass or atomic number,
- protein vs ligand entity embedding.

Recommended additional chemistry flags:

- H-bond donor,
- H-bond acceptor,
- cationic,
- anionic,
- hydrophobic.

### 5.2 Topological positional encoding

Retain random-walk positional encoding as a static topology anchor.

Use the sparse CSR implementation for preprocessing rather than dense adjacency powers.

Default:

```yaml
rwpe_dim: 20
```

The PE should remain static through recurrence.

### 5.3 Intra-edge features

Recommended components:

```text
bond type
conjugation
ring membership
aromatic bond flag
covalent/spatial relation type
distance RBF
optional shortest-path / ring-system indicator
```

Intra-edge features should normally remain static.

### 5.4 Interface-edge raw features

Recommended initial interface representation `Q^0`:

```text
distance RBF
H-bond donor/acceptor compatibility
opposite/same charge indicators
hydrophobic compatibility
protein local-orientation cosine
ligand local-orientation cosine
optional atom-pair type embedding
```

The original physical/chemical feature remains available at every recurrent cycle.

---

## 6. Geometry and invariance

The predicted affinity is a scalar and must be invariant to global translation and rotation.

At minimum, all geometry injected into the scalar network must therefore be based on E(3)-invariant quantities such as

```text
||r_ij||
RBF(||r_ij||)
unit-vector dot products
angles
vector norms
inner products of equivariant features
```

Do not feed absolute Cartesian coordinates directly into an unconstrained MLP.

### 6.1 Local orientation

For atom `i`, estimate a local outward direction from covalent neighbors:

```text
u_i = normalize(x_i - mean_{k in N_cov(i)} x_k)
```

For interface pair `(i,j)` with

```text
r_ij = x_j - x_i
rhat_ij = normalize(r_ij)
```

add

```text
cos_P = u_i · rhat_ij
cos_L = u_j · (-rhat_ij)
```

These provide directional information while remaining globally rotation invariant.

---

## 7. Optional equivariant hidden-state extension

Equivariance should be treated as an extension of the recurrent interaction engine, not as a requirement to move coordinates.

Recommended configuration:

```yaml
model:
  equivariant:
    enabled: false
    vector_channels: 32
    interface_only: true
    coordinate_update: false
```

### 7.1 Scalar + vector state

Represent each node as

```text
(h_i, V_i)
```

where

```text
h_i ∈ R^d
V_i ∈ R^(d_v × 3)
```

`V_i` transforms equivariantly under 3D rotation.

### 7.2 Equivariant edge message

For relative unit vector `rhat_ij`, define a scalar message

```text
m_ij^s = phi_s(h_i, h_j, e_ij, RBF(d_ij))
```

and vector message

```text
m_ij^v = phi_v(m_ij^s) ⊙ rhat_ij
```

After aggregation,

```text
M_i^v = Σ_j eta_ij m_ij^v
```

convert vector information back to invariant scalar quantities using channel-wise norms and/or invariant contractions before updating scalar states.

### 7.3 Geometry-aware gating

A graph edge gate may depend on invariant projections such as

```text
<V_i, rhat_ij>
<V_j, rhat_ij>
||V_i||
||V_j||
```

The gate itself must remain invariant.

### 7.4 Coordinate update policy

Default:

```yaml
coordinate_update: false
```

Rationale: BA-Pred2 predicts affinity from a supplied pose. Updating coordinates would mix pose relaxation with affinity prediction and introduces an underconstrained latent coordinate transformation.

Coordinate refinement belongs in a separate BA-Pred2/RMSD-Pred or pose-refinement experiment.

---

## 8. Encoders and prelude

Map raw features to hidden dimension `d`.

Recommended starting point:

```yaml
hidden_dim: 256
```

Node initialization:

```text
H_P^0 = LN(W_P x_P + W_pe p_P + entity_P)
H_L^0 = LN(W_L x_L + W_pe p_L + entity_L)
```

Intra edges:

```text
Z_P^0 = phi_edge_P(e_P)
Z_L^0 = phi_edge_L(e_L)
```

Interface:

```text
Q^0 = phi_interface(e_PL)
```

Use one or two non-recurrent local GNN layers before the recurrent core so that cross interaction starts from chemically contextualized atom states.

The prelude should be independently ablated; it must not become another deep stack.

---

## 9. Recurrent interaction block

### 9.1 Interface update

For protein atom `i` and ligand atom `j`, build

```text
u_ij^t = concat(
    LN(h_i^t),
    LN(h_j^t),
    LN(q_ij^t),
    h_i^0,
    h_j^0,
    q_ij^0,
    geometry_ij
)
```

Compute candidate interface state

```text
qhat_ij^(t+1) = phi_q(u_ij^t)
```

and gate

```text
g_ij^q = sigmoid(phi_gq(u_ij^t))
```

Then update by interpolation:

```text
q_ij^(t+1) = (1 - g_ij^q) ⊙ q_ij^t + g_ij^q ⊙ qhat_ij^(t+1)
```

This is preferred over unrestricted positive residual accumulation.

### 9.2 Optional sparse interface-triangle update

This is a planned v0.2+ feature rather than a mandatory first baseline.

For a contact `(p,l)`, aggregate compatible neighboring contacts sharing one endpoint.

Protein-shared motif:

```text
(p,l1), (p,l2), and ligand relation (l1,l2)
```

Ligand-shared motif:

```text
(p1,l), (p2,l), and protein relation (p1,p2)
```

Conceptually:

```text
T_PL-L(p,l) = Σ_k phi(q_pl, q_pk, z_lk)
T_PP-L(p,l) = Σ_k phi(q_pl, q_kl, z_pk)
```

These summaries may be injected into `phi_q`.

Do not build a dense N×N pair tensor. Preserve sparse scaling.

### 9.3 Cross messages

Protein-to-ligand:

```text
m_i→j = phi_PL(h_i^t, q_ij^(t+1))
```

Gate/importance:

```text
a_ij = phi_a(h_i^t, h_j^t, q_ij^(t+1))
```

Normalize over incoming interface neighbors:

```text
alpha_ij = sigmoid(a_ij) / (Σ_k sigmoid(a_kj) + eps)
```

Aggregate:

```text
M_j^PL = Σ_i alpha_ij m_i→j
```

Build the reverse message analogously for ligand-to-protein.

Cross-conditioned update:

```text
Delta h_j^cross = phi_cross(LN(h_j^t), M_j^PL, h_j^0)
g_j^cross = sigmoid(phi_cross_gate(...))
hbar_j^t = h_j^t + lambda_cross * g_j^cross ⊙ Delta h_j^cross
```

Use a small LayerScale initialization, e.g. 0.05–0.2.

### 9.4 Intra-molecular propagation

For an intra edge `(i,j)`:

```text
a_ij^intra = phi_a(LN(hbar_i), LN(hbar_j), z_ij^0)
eta_ij = sigmoid(a_ij^intra) / (Σ_k sigmoid(a_kj^intra) + eps)
m_ij = eta_ij * phi_v(hbar_i, z_ij^0)
M_j = Σ_i m_ij
```

Candidate update:

```text
Delta h_j^intra = phi_intra(LN(hbar_j), M_j, h_j^0)
g_j^intra = sigmoid(phi_intra_gate(...))
h_j^(t+1) = hbar_j^t + lambda_intra * g_j^intra ⊙ Delta h_j^intra
```

Protein and ligand intra operators should initially use separate parameters.

A later ablation may test parameter sharing.

---

## 10. Normalization and recurrent stability

Inside the recurrent core:

- use LayerNorm or RMSNorm,
- prefer pre-norm updates,
- initialize LayerScale/residual gain small,
- avoid BatchNorm,
- avoid permanently growing edge states,
- clip gradients,
- log recurrent-state norms.

Recommended diagnostics per recycle step:

```text
mean ||H_P^(t+1) - H_P^t||
mean ||H_L^(t+1) - H_L^t||
mean ||Q^(t+1) - Q^t||
mean/std ||H_P^t||
mean/std ||H_L^t||
mean/std ||Q^t||
mean pairwise node cosine similarity
optional graph Dirichlet energy
```

The purpose is to detect oversmoothing, oscillation, saturation, or exploding recurrent dynamics.

---

## 11. Readout

Do not discard the learned interface state.

Recommended pooled representations:

```text
H_L_global       = sum/attention-pool ligand atoms
H_P_contact      = contact-weighted pool of protein atoms
H_L_contact      = contact-weighted pool of ligand atoms
H_interface      = weighted pool of Q^T
```

Affinity head:

```text
y_hat = MLP(concat(H_L_global, H_P_contact, H_L_contact, H_interface))
```

Output is predicted `pKd`.

Start with sum pooling for direct comparability to BA-Pred, then ablate learned attention pooling.

---

## 12. Recycle training

Training must not always use a single fixed number of recurrent steps.

Initial recycle support:

```yaml
train_recycles: [2, 3, 4, 6, 8]
```

Sample one recycle count per batch or per example, depending on implementation efficiency.

A reasonable initial probability distribution should favor shorter paths while still exposing the model to long paths.

Example:

```text
P(T=2)=0.20
P(T=3)=0.25
P(T=4)=0.25
P(T=6)=0.20
P(T=8)=0.10
```

The model should learn an iterative operator rather than encode absolute layer identity.

Future scaling experiment:

- train with maximum T=8,
- evaluate at T ∈ {1,2,3,4,6,8,12,16}.

A successful recurrent model should degrade gracefully and ideally continue improving beyond the mean training depth.

---

## 13. Loss and optimization

Primary regression target: `pKd`.

Recommended initial loss:

```yaml
loss: huber
```

Also support:

```text
MSE
MAE
```

Recommended optimizer baseline:

```yaml
optimizer: AdamW
learning_rate: 1e-4
weight_decay: 1e-5
grad_clip_norm: 5.0
scheduler: cosine
```

Use mixed precision when available.

Track at least:

- RMSE,
- MAE,
- Pearson r,
- Spearman rho.

Checkpoint on validation RMSE unless an experiment specifies otherwise.

---

## 14. Dataset preprocessing

### 14.1 Expected manifest schema

The generic preprocessing entry point should accept a CSV with at least:

```text
id,protein_path,ligand_path,pKd,split
```

Optional columns may include target family, cluster IDs, release year, source dataset, or benchmark membership.

### 14.2 PDBbind helper

Provide a helper that converts a standard PDBbind directory/index layout into the generic manifest.

Expected search targets include common names such as

```text
*_protein.pdb
*_pocket.pdb
*_ligand.sdf
*_ligand.mol2
```

Do not hard-code random split as the only supported evaluation setting.

### 14.3 Pocket extraction

Default BA-Pred-like pocket definition:

```yaml
pocket_cutoff: 8.0
```

Include all atoms belonging to residues that have at least one atom within the cutoff from any ligand atom.

Keep the cutoff configurable.

### 14.4 Cached graph format

Preprocessing should save graph objects to disk so training performs no RDKit/PDB parsing.

The processed manifest should reference cached graph files.

Store preprocessing metadata/config with each dataset cache or alongside the processed manifest so that graph-generation settings are reproducible.

---

## 15. Data splits

Random split may be used for smoke tests and direct compatibility experiments, but it must not be the only reported result.

Target evaluation plan:

1. BA-Pred-compatible/random split,
2. protein-cluster split,
3. ligand-scaffold split,
4. combined protein/ligand OOD split where feasible,
5. temporal split where dataset metadata permits,
6. external benchmarks if compatible.

The recurrent design is especially interesting if it improves OOD/generalization rather than only random-split fitting.

---

## 16. Evaluation protocol

For each trained checkpoint, evaluate the exact same weights across multiple inference recycle counts.

Recommended sweep:

```text
T = 1, 2, 3, 4, 6, 8, 12, 16
```

Report:

```text
RMSE(T)
MAE(T)
Pearson(T)
Spearman(T)
inference cost(T)
state-delta diagnostics(T)
```

The key recurrent-model figure should be performance vs test-time recycle count.

Interpretation:

```text
performance improves/plateaus with increasing T
    → iterative refinement behavior

performance collapses immediately after the training depth
    → likely learned an unrolled-depth schedule rather than a reusable inference operator
```

---

## 17. Required baselines and ablations

Do not change every architectural component at once and claim the resulting difference is due to recurrence.

Recommended staged experimental sequence:

### Stage A — current/reference baseline

- Original BA-Pred or closest reproducible PyG baseline.

### Stage B — clean scalar graph baseline

- remove broken/unused MHA,
- LayerNorm-based GNN,
- no recurrence,
- retain approximately similar effective depth.

### Stage C — scalar recurrent BA-Pred2

- static input anchoring,
- gated recurrent updates,
- dynamic interface state,
- no equivariant vector state.

### Stage D — graph enrichment

Individually test:

- + protein spatial edges,
- + ligand spatial edges,
- + local orientation features,
- + richer interface chemistry.

### Stage E — interface topology

- + sparse shared-endpoint contact context,
- + explicit P-P-L / P-L-L triangle update.

### Stage F — equivariance

Compare:

```text
scalar recurrent
scalar + equivariant interface only
scalar + equivariant all-graph
```

Coordinate updates remain disabled.

### Stage G — recycle scaling

Train with variable T and evaluate outside the training-depth distribution.

---

## 18. Model-size and compute reporting

Every reported experiment should record:

```text
parameter count
mean training recycle count
inference recycle count
estimated FLOPs or wall time
peak GPU memory
```

The recurrent architecture should be judged on quality vs compute and quality vs parameter count, not accuracy alone.

---

## 19. Training CLI requirements

The project should expose a reproducible flow equivalent to

```bash
# build a manifest
python scripts/make_pdbbind_manifest.py \
  --root /path/to/PDBbind \
  --index /path/to/INDEX_general_PL_data \
  --out data/pdbbind.csv

# preprocess graphs
bapred2-preprocess \
  --manifest data/pdbbind.csv \
  --out data/processed \
  --config configs/bapred2_base.yaml

# train
bapred2-train \
  --manifest data/processed/processed_manifest.csv \
  --config configs/bapred2_base.yaml \
  --out runs/base \
  --device cuda

# recycle sweep
bapred2-eval \
  --manifest data/processed/processed_manifest.csv \
  --checkpoint runs/base/best.pt \
  --split test \
  --recycles 1,2,3,4,6,8,12,16
```

All important model/preprocessing values must live in YAML configuration rather than being scattered through Python source.

---

## 20. Recommended configuration structure

```yaml
seed: 42

data:
  pocket_cutoff: 8.0
  interface_cutoff: 5.0
  protein_spatial_cutoff: 5.0
  ligand_spatial_cutoff: 4.5
  protein_max_spatial_neighbors: 32
  ligand_max_spatial_neighbors: 24
  distance_rbf_dim: 16
  rwpe_dim: 20

model:
  hidden_dim: 256
  prelude_layers: 1
  dropout: 0.1
  layerscale_init: 0.1
  train_recycles: [2, 3, 4, 6, 8]

  interface_triangle:
    enabled: false

  equivariant:
    enabled: false
    vector_channels: 32
    interface_only: true
    coordinate_update: false

train:
  batch_size: 16
  epochs: 200
  lr: 1.0e-4
  weight_decay: 1.0e-5
  grad_clip: 5.0
  loss: huber
  amp: true
  early_stopping_patience: 25
```

These values are experiment defaults, not fixed scientific assumptions.

---

## 21. Testing requirements

At minimum, CI should test:

1. graph preprocessing on a tiny synthetic/example complex,
2. no NaN/Inf in encoded graph features,
3. model forward at multiple recycle counts,
4. backward pass through the recurrent model,
5. invariant scalar output under rigid rotation/translation of an example graph when raw graph features are recomputed,
6. equivariant vector-feature rotation test when the equivariant mode is enabled,
7. batch-size >1 heterograph handling,
8. empty/low-contact interface handling,
9. checkpoint save/load reproducibility.

The rigid-transform tests are required before labeling an implementation E(3)-invariant/equivariant.

---

## 22. Oversmoothing and stability experiments

Since recurrent GNNs can oversmooth, explicitly measure it.

For each recycle depth, record one or more of:

```text
mean inter-node cosine similarity
feature variance across nodes
Dirichlet energy over graph edges
rank/effective-rank of H
state-update magnitude
```

Plot these against recycle depth together with predictive performance.

This allows distinguishing useful iterative convergence from representational collapse.

---

## 23. What not to do in the first version

Avoid the following until the scalar recurrent baseline is established:

- dense all-pairs pair representation,
- full Pairformer-style O(N²) tensor,
- coordinate refinement,
- diffusion/flow-based pose updates,
- very deep non-shared prelude layers,
- simultaneous introduction of triangle updates, equivariance, new losses, and new datasets without ablations.

BA-Pred2 should remain a sparse, scalable protein–ligand graph model.

---

## 24. Main scientific questions

The project should be able to answer the following clearly.

### Q1. Can recurrent graph inference replace independently parameterized depth?

Compare equal effective depth and equal/near-equal compute while reporting parameter count.

### Q2. Does extra test-time recurrence improve affinity prediction?

Evaluate a fixed checkpoint over a broad recycle sweep.

### Q3. Does input anchoring prevent oversmoothing and recurrent drift?

Ablate `H^0/Q^0` reinjection.

### Q4. Is a dynamic interface representation more useful than repeatedly updating generic intra edges?

Ablate dynamic `Q` vs static interface embedding.

### Q5. Do protein/ligand spatial edges improve OOD behavior?

Evaluate beyond random splits.

### Q6. Does sparse contact topology capture cooperative binding geometry?

Test shared-endpoint context and explicit interface triangles separately.

### Q7. Does equivariant directional state add value beyond invariant handcrafted geometry?

Compare scalar recurrent vs interface-only equivariant vs fully equivariant hidden-state variants.

---

## 25. Success criteria

BA-Pred2 is scientifically successful if at least some of the following are demonstrated reproducibly:

1. equal or better affinity metrics with substantially fewer parameters,
2. improved protein/ligand OOD performance,
3. useful test-time compute scaling with increasing recycle count,
4. improved robustness to difficult pocket/contact environments,
5. stable recurrent dynamics without strong oversmoothing,
6. measurable gain from geometric/equivariant interface reasoning,
7. interpretable evolution of interface-contact importance through recycle steps.

A small random-split RMSE gain alone is not sufficient to validate the architecture.

---

## 26. Immediate implementation milestones

### Milestone 0 — runnable scalar baseline

- end-to-end preprocessing,
- train/eval CLI,
- scalar recurrent block,
- dynamic interface state,
- static anchors,
- spatial edges,
- recycle sweep,
- unit/smoke tests.

### Milestone 1 — rigorous baseline reproduction

- reproduce original BA-Pred split and metrics as closely as possible,
- establish controlled non-recurrent BA-Pred2 baseline,
- profile memory/runtime/parameters.

### Milestone 2 — recurrent analysis

- variable recycle training,
- test-time recycle scaling,
- oversmoothing/state-dynamics diagnostics,
- fixed-depth vs recurrent comparison.

### Milestone 3 — sparse interface geometry

- shared-endpoint contact context,
- P-P-L and P-L-L sparse triangle update,
- explicit ablation.

### Milestone 4 — equivariant gating

- scalar/vector hidden state,
- geometry-aware invariant gates,
- interface-only mode first,
- rigid-transform tests,
- full-graph equivariant mode only if justified.

### Milestone 5 — OOD and benchmark evaluation

- protein cluster split,
- ligand scaffold split,
- combined OOD evaluation,
- external affinity benchmarks where compatible.

---

## 27. Short architecture summary

BA-Pred2 should ultimately be describable as:

> A sparse recurrent protein–ligand graph neural network that repeatedly refines a dynamic interface representation, propagates the resulting binding context through protein and ligand molecular graphs, and optionally maintains E(3)-equivariant directional hidden states while keeping coordinates fixed. Immutable molecular and geometric input features are re-injected at every recycle step to stabilize deep recurrent inference and reduce oversmoothing.

That sentence defines the intended architecture. Implementations that materially deviate from it should be treated as separate experiments rather than silently replacing the baseline.
