# KGR-Survey
A Survey of Task-Oriented Knowledge Graph Reasoning: Status, Applications, and Prospects

## Survey Paper
| Title   | Conference/Journal | Year   | Characteristic | Paper |
| ------- | :------: | :-------: | :-------: | :-------: |
| Knowledge graph embedding: a survey from the perspective of representation spaces | ACM Computer Survey | 2024  | Embedding Spaces| [link](https://dl.acm.org/doi/abs/10.1145/3643806)
| A survey of knowledge graph reasoning on graph types: Static, dynamic, and multi-modal | IEEE TPAMI | 2024 | Graph Types | [link](https://ieeexplore.ieee.org/document/10577554)
| Negative sampling in knowledge graph representation learning: a review | arXiv | 2024 | Negative Sampling | [link](https://arxiv.org/abs/2402.19195)
| Overview of knowledge reasoning for knowledge graph | Neurocomputering | 2024 | Causal Reasoning | [link](https://www.sciencedirect.com/science/article/abs/pii/S0925231224003424)
| A survey on temporal knowledge graph: representation learning and applications | arXiv | 2024 | Temporal Reasoning | [link](https://arxiv.org/abs/2403.04782)
| A survey on temporal knowledge graph completion: taxonomy, progress, and prospects | arXiv | 2023 | Temporal Reasoning | [link](https://arxiv.org/abs/2308.02457)
| Generalizing to unseen elements: a survey on knowledge extrapolation for knowledge graphs | IJCAI | 2023 | Unseen Elements |[link](https://www.ijcai.org/proceedings/2023/737)
| A survey on few-shot knowledge graph completion with structural and commonsense knowledge | arXiv | 2023 | Commonsense | [link](https://arxiv.org/abs/2301.01172)
| Beyond transduction: a survey on inductive, few shot, and zero shot link prediction in knowledge graphs | arXiv | 2023 | Few-shot & Inductive | [link](https://arxiv.org/abs/2312.04997)
| A comprehensive overview of knowledge graph completion | Knowledge-Based System | 2022 | Multi-modal & Hyper-relation | [link](https://www.sciencedirect.com/science/article/abs/pii/S095070512200805X)
| Knowledgegraph reasoning with logics and embeddings: survey and perspective | arXiv | 2022 | Logics and Embeddings | [link](https://arxiv.org/pdf/2202.07412)

## Static Single-Step KGR
### KGE-based KGR Model

<div align="center">
  <h3><strong>Translation or Tensor Decomposition-Based KGE Models</strong></h3>
</div>

| Model  | Title | Conference/Journal | Year | Paper |
|:-----:|---------------------------------|:---------------------------------:|:------:|:------:|
| TransE | Translating embeddings for modeling multi-relational data | NIPS | 2013 | [link](https://proceedings.neurips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html) |
| TransH | Knowledge graph embedding by translating on hyperplanes | AAAI | 2014 | [link](https://ojs.aaai.org/index.php/AAAI/article/view/8870) |
| TransR | Learning entity and relation embeddings for knowledge graph completion | AAAI | 2015 | [link](https://aaai.org/papers/9491-learning-entity-and-relation-embeddings-for-knowledge-graph-completion/) |
| TransD | Knowledge graph embedding via dynamic mapping matrix | ACL | 2015 | [doi](https://doi.org/10.3115/v1/P15-1067) |
| TranSparse | Knowledge graph completion with adaptive sparse transfer matrix | AAAI | 2016 | [link](https://aaai.org/papers/10089-knowledge-graph-completion-with-adaptive-sparse-transfer-matrix/) |
| PairE | PairRE: Knowledge graph embeddings via paired relation vectors | ACL | 2021 | [doi](https://doi.org/10.18653/v1/2021.acl-long.336) |
| TransA | TransA: An adaptive approach for knowledge graph embedding | arXiv | 2015 | [arXiv](https://arxiv.org/abs/1509.05490) |
| KG2E | Learning to represent knowledge graphs with Gaussian embedding | CIKM | 2015 | [doi](https://doi.org/10.1145/2806416.2806502) |
| ManifoldE | From one point to a manifold: Knowledge graph embedding for precise link prediction | IJCAI | 2016 | [link](https://www.ijcai.org/Proceedings/16/Papers/190.pdf) |
| TorusE | TorusE: Knowledge graph embedding on a Lie group | AAAI | 2018 | [link](https://ojs.aaai.org/index.php/AAAI/article/view/11538) |
| Poincaré | Poincare embeddings for learning hierarchical representations | NIPS | 2017 | [link](https://papers.nips.cc/paper_files/paper/2017/hash/59dfa2df42d9e3d41f5b02bfc32229dd-Abstract.html) |
| MuRP | Multi-relational Poincare graph embeddings | NIPS | 2019 | [link](https://proceedings.neurips.cc/paper_files/paper/2019/file/f8b932c70d0b2e6bf071729a4fa68dfc-Paper.pdf) |
| HAKE | Learning hierarchy-aware knowledge graph embeddings for link prediction | AAAI | 2020 | [doi](https://doi.org/10.1609/aaai.v34i03.5701) |
| H2E | Knowledge graph representation via hierarchical hyperbolic neural graph embedding | IEEE Big Data | 2021 | [doi](https://doi.org/10.1109/BigData52589.2021.9671651) |
| HBE | Hyperbolic hierarchy-aware knowledge graph embedding for link prediction | EMNLP | 2021 | [doi](https://doi.org/10.18653/v1/2021.findings-emnlp.251) |
| RotatE | RotatE: Knowledge graph embedding by relational rotation in complex space | ICLR | 2019 | [link](https://arxiv.org/pdf/1902.10197) |
| QuatE | Quaternion knowledge graph embedding | NIPS | 2019 | [link](https://proceedings.neurips.cc/paper_files/paper/2019/file/d961e9f236177d65d21100592edb0769-Paper.pdf) |
| DualE | Dual quaternion knowledge graph embeddings | AAAI | 2021 | [doi](https://doi.org/10.1609/aaai.v35i8.16850) |
| RESCAL | A three-way model for collective learning on multi-relational data | ICML | 2011 | [link](https://icml.cc/2011/papers/438_icmlpaper.pdf) |
| PITF-BPR | Predicting RDF triples in incomplete knowledge bases with tensor factorization | SAC | 2012 | [link](https://doi.org/10.1007/978-3-319-25007-6_37) |
| DistMult | Embedding entities and relations for learning and inference in knowledge bases | ICLR | 2015 | [link](https://arxiv.org/pdf/1412.6575) |
| ComplEx | Complex embeddings for simple link prediction | ICML | 2016 | [link](https://arxiv.org/pdf/1606.06357) |
| HolE | Holographic embeddings of knowledge graphs | AAAI | 2016 | [link](https://ojs.aaai.org/index.php/AAAI/article/view/10314) |
