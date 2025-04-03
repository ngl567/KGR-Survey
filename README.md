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
| TransD | Knowledge graph embedding via dynamic mapping matrix | ACL | 2015 | [link](https://doi.org/10.3115/v1/P15-1067) |
| TranSparse | Knowledge graph completion with adaptive sparse transfer matrix | AAAI | 2016 | [link](https://aaai.org/papers/10089-knowledge-graph-completion-with-adaptive-sparse-transfer-matrix/) |
| PairE | PairRE: Knowledge graph embeddings via paired relation vectors | ACL | 2021 | [link](https://doi.org/10.18653/v1/2021.acl-long.336) |
| TransA | TransA: An adaptive approach for knowledge graph embedding | arXiv | 2015 | [link](https://arxiv.org/abs/1509.05490) |
| KG2E | Learning to represent knowledge graphs with Gaussian embedding | CIKM | 2015 | [link](https://doi.org/10.1145/2806416.2806502) |
| ManifoldE | From one point to a manifold: Knowledge graph embedding for precise link prediction | IJCAI | 2016 | [link](https://www.ijcai.org/Proceedings/16/Papers/190.pdf) |
| TorusE | TorusE: Knowledge graph embedding on a Lie group | AAAI | 2018 | [link](https://ojs.aaai.org/index.php/AAAI/article/view/11538) |
| Poincaré | Poincare embeddings for learning hierarchical representations | NIPS | 2017 | [link](https://papers.nips.cc/paper_files/paper/2017/hash/59dfa2df42d9e3d41f5b02bfc32229dd-Abstract.html) |
| MuRP | Multi-relational Poincare graph embeddings | NIPS | 2019 | [link](https://proceedings.neurips.cc/paper_files/paper/2019/file/f8b932c70d0b2e6bf071729a4fa68dfc-Paper.pdf) |
| HAKE | Learning hierarchy-aware knowledge graph embeddings for link prediction | AAAI | 2020 | [link](https://doi.org/10.1609/aaai.v34i03.5701) |
| H2E | Knowledge graph representation via hierarchical hyperbolic neural graph embedding | IEEE Big Data | 2021 | [link](https://doi.org/10.1109/BigData52589.2021.9671651) |
| HBE | Hyperbolic hierarchy-aware knowledge graph embedding for link prediction | EMNLP | 2021 | [link](https://doi.org/10.18653/v1/2021.findings-emnlp.251) |
| RotatE | RotatE: Knowledge graph embedding by relational rotation in complex space | ICLR | 2019 | [link](https://arxiv.org/pdf/1902.10197) |
| QuatE | Quaternion knowledge graph embedding | NIPS | 2019 | [link](https://proceedings.neurips.cc/paper_files/paper/2019/file/d961e9f236177d65d21100592edb0769-Paper.pdf) |
| DualE | Dual quaternion knowledge graph embeddings | AAAI | 2021 | [link](https://doi.org/10.1609/aaai.v35i8.16850) |
| RESCAL | A three-way model for collective learning on multi-relational data | ICML | 2011 | [link](https://icml.cc/2011/papers/438_icmlpaper.pdf) |
| PITF-BPR | Predicting RDF triples in incomplete knowledge bases with tensor factorization | SAC | 2012 | [link](https://doi.org/10.1007/978-3-319-25007-6_37) |
| DistMult | Embedding entities and relations for learning and inference in knowledge bases | ICLR | 2015 | [link](https://arxiv.org/pdf/1412.6575) |
| ComplEx | Complex embeddings for simple link prediction | ICML | 2016 | [link](https://arxiv.org/pdf/1606.06357) |
| HolE | Holographic embeddings of knowledge graphs | AAAI | 2016 | [link](https://ojs.aaai.org/index.php/AAAI/article/view/10314) |

<div align="center">
  <h3><strong>(Graph) Neural Network-based Models</strong></h3>
</div>

| Model  | Title | Conference/Journal | Year | Paper |
|:-----:|---------------------------------|:---------------------------------:|:------:|:------:|
| NTN  | Reasoning with neural tensor networks for knowledge base completion | NIPS | 2013 | [link](https://proceedings.neurips.cc/paper_files/paper/2013/hash/b337e84de8752b27eda3a12363109e80-Abstract.html) |
| SME | A semantic matching energy function for learning with multi-relational data | Machine Learning | 2014 | [link](https://doi.org/10.1007/s10994-013-5363-6) |
| NAM | Probabilistic reasoning via deep learning: Neural association models | arXiv | 2016 | [link](https://arxiv.org/abs/1603.07704) |
| ConvE | Convolutional 2D knowledge graph embeddings | AAAI | 2018 | [link](https://ojs.aaai.org/index.php/AAAI/article/view/11573) |
| ConvKB | A novel embedding model for knowledge base completion based on convolutional neural network | NAACL | 2018 | [link](https://doi.org/10.18653/v1/N18-2053) |
| GNN Survey | A comprehensive survey on graph neural networks | IEEE TNNLS | 2021 | [link](https://doi.org/10.1109/TNNLS.2020.2978386) |
| R-GCN | Modeling relational data with graph convolutional networks | ESWC | 2018 | [Link](https://doi.org/10.1007/978-3-319-93417-4_38) |
| SACN | End-to-end structure-aware convolutional networks for knowledge base completion | AAAI | 2019 | [link](https://ojs.aaai.org/index.php/AAAI/article/view/4164) |
| KBGAT | Learning attention-based embeddings for relation prediction in knowledge graphs | ACL | 2019 | [link](https://doi.org/10.18653/v1/P19-1466) |
| KE-GCN | Knowledge embedding based graph convolutional network | The Web Conference | 2021 | [link](https://doi.org/10.1145/3442381.3449925) |

<div align="center">
  <h3><strong>Transformer-based Models</strong></h3>
</div>

| Model  | Title | Conference/Journal | Year | Paper |
|:-----:|---------------------------------|:---------------------------------:|:------:|:------:|
| KG-BERT | Modeling relational data with graph convolutional networks | ESWC | 2018 | [Link](https://doi.org/10.1007/978-3-319-93417-4_38) |
| R-MeN | A relational memory-based embedding model for triple classification and search personalization | ACL | 2021 | [link](https://aclanthology.org/2020.acl-main.313/) |
| CoKE | CoKE: Contextualized knowledge graph embedding | arXiv | 2019 | [link](https://arxiv.org/abs/1911.02168) |
| HittER | HittER: Hierarchical transformers for knowledge graph embeddings | EMNLP | 2021 | [link](https://aclanthology.org/2021.emnlp-main.812/) |
| GenKGC | From discrimination to generation: Knowledge graph completion with generative transformer | WWW | 2022 | [link](https://doi.org/10.1145/3487553.3524238) |
| iHT | Pre-training transformers for knowledge graph completion | arXiv | 2023 | [link](https://arxiv.org/abs/2303.15682) |
| SimKGC | SimKGC: Simple contrastive knowledge graph completion with pre-trained language models | ACL | 2022 | [link](https://doi.org/10.18653/v1/2022.acl-long.295) |
| StAR | Structure-augmented text representation learning for efficient knowledge graph completion | WWW | 2021 | [link](https://doi.org/10.1145/3442381.3450043) |
| KoPA | Making large language models perform better in knowledge graph completion | arXiv | 2023 | [link](https://arxiv.org/abs/2310.06671) |
| KICGPT | KICGPT: Large language model with knowledge in context for knowledge graph completion | EMNLP | 2023 | [link](https://doi.org/10.18653/v1/2023.findings-emnlp.580) |
| Relphormer | Relphormer: Relational graph transformer for knowledge graph representations | Neurocomputing | 2024 | [link](https://doi.org/10.1016/j.neucom.2023.127044) |
| LGKGR | LGKGR: A knowledge graph reasoning model using LLMs augmented GNNs | Neurocomputing | 2025 | [Link](https://doi.org/10.1016/j.neucom.2025.129919) |

<div align="center">
  <h3><strong>Ontology-Enhanced KGE Model</strong></h3>
</div>

| Model  | Title | Conference/Journal | Year | Paper |
|:-----:|---------------------------------|:---------------------------------:|:------:|:------:|
| JOIE  | Universal representation learning of knowledge bases by jointly embedding instances and ontological concepts | KDD | 2019 | [Link](https://doi.org/10.1145/3292500.3330838) |
| Nickel et al. | Factorizing YAGO: Scalable machine learning for linked data | WWW | 2012 | [link](https://doi.org/10.1145/2187836.2187874) |
| CISS | Embedding two-view knowledge graphs with class inheritance and structural similarity | KDD | 2024 | [link](https://doi.org/10.1145/3637528.3671941) |
| Wang et al. | An ontology-enhanced knowledge graph embedding method | ICCPR | 2024 | [link](https://doi.org/10.1145/3633637.3633645) |
| Concept2Box | Concept2Box: Joint geometric embeddings for learning two-view knowledge graphs | ACL | 2023 | [link](https://doi.org/10.18653/v1/2023.findings-acl.642) |
| CAKE | CAKE: A scalable commonsense-aware framework for multi-view knowledge graph completion | ACL | 2022 | [link](https://doi.org/10.18653/v1/2022.acl-long.205) |
| SSE | Semantically smooth knowledge graph embedding | ACL | 2015 | [link](https://doi.org/10.3115/v1/P15-1009) |
| TKRL | Representation learning of knowledge graphs with hierarchical types | IJCAI | 2016 | [link](https://www.ijcai.org/Proceedings/16/Papers/421.pdf) |
| TransET | TransET: Knowledge graph embedding with entity types | Electronics | 2021 | [link](https://doi.org/10.3390/electronics10121407) |
| AutoETER | AutoETER: Automated entity type representation for knowledge graph embedding | EMNLP | 2020 | [link](https://doi.org/10.18653/v1/2020.findings-emnlp.105) |

<div align="center">
  <h3><strong>Path-Enhanced KGE Model</strong></h3>
</div>

| Model  | Title | Conference/Journal | Year | Paper |
|:-----:|---------------------------------|:---------------------------------:|:------:|:------:|
| Path-RNN | Compositional vector space models for knowledge base completion | ACL | 2015 | [link](https://doi.org/10.3115/v1/P15-1016) |
| PTransE  | Modeling relation paths for representation learning of knowledge bases | EMNLP | 2015 | [Link](https://doi.org/10.18653/v1/D15-1082) |
| PRN | A path-based relation networks model for knowledge graph completion | Expert Systems with Applications | 2021 | [link](https://doi.org/10.1016/j.eswa.2021.115273) |
| OPTransE | Representation learning with ordered relation paths for knowledge graph completion | EMNLP-IJCNLP | 2019 | [link](https://doi.org/10.18653/v1/D19-1268) |
| TransE\&RW | Modeling relation paths for knowledge base completion via joint adversarial training | Knowledge Based Systems | 2020 | [link](https://doi.org/10.1016/j.knosys.2020.105865) |
| HARPA | HARPA: hierarchical attention with relation paths for knowledge graph embedding adversarial learning | Data Mining and Knowledge Discovery | 2023 | [link](https://doi.org/10.1007/s10618-022-00888-3) |
| RPJE | Rule-guided compositional representation learning on knowledge graphs | AAAI | 2020 | [link](https://doi.org/10.1609/aaai.v34i03.5687) |
| PARL | Attention-aware path-based relation extraction for medical knowledge graph | Smart Computing and Communication | 2017 | [link](https://doi.org/10.1007/978-3-319-73830-7_32) |
| Das et al. | Chains of reasoning over entities, relations, and text using recurrent neural networks | EACL | 2017 | [link](https://aclanthology.org/E17-1013/) |
| Jiang et al. | Attentive path combination for knowledge graph completion | Machine Learning Research | 2017 | [link](https://proceedings.mlr.press/v77/jiang17a.html) |
| CPConvKE | A confidence-aware and path-enhanced convolutional neural network embedding framework on noisy knowledge graph | Neurocomputing | 2023 | [link](https://doi.org/10.1016/j.neucom.2023.126261) |
| PaSKoGE | Path-specific knowledge graph embedding | Knowledge-based Systems | 2018 | [link](https://doi.org/10.1016/j.knosys.2018.03.020) |
| Jagvaral et al. | Path-based reasoning approach for knowledge graph completion using CNN-BiLSTM with attention mechanism | Expert Systems with Applications | 2020 | [link](https://doi.org/10.1016/j.eswa.2019.112960) |
| PathCon | Relational message passing for knowledge graph completion | KDD | 2021 | [link](https://doi.org/10.1145/3447548.3467247) |
| PTrustE | PTrustE: A high-accuracy knowledge graph noise detection method based on path trustworthiness and triple embedding | Knowledge-based Systems | 2022 | [link](https://doi.org/10.1016/j.knosys.2022.109688) |
| TAPR | Modeling relation paths for knowledge graph completion | IEEE TKDE | 2021 | [link](https://doi.org/10.1109/TKDE.2020.2970044) |
| Niu et al. | Joint semantics and data-driven path representation for knowledge graph reasoning | Neurocomputing | 2022 | [link](https://doi.org/10.1016/j.neucom.2022.02.011) |

<div align="center">
  <h3><strong>Negative Sampling for KGE</strong></h3>
</div>

| Model  | Title | Conference/Journal | Year | Paper |
|:-----:|---------------------------------|:---------------------------------:|:------:|:------:|
| Local Closed-World Assumption | Knowledge Vault: A web scale approach to probabilistic knowledge fusion | KDD | 2014 | [link](https://doi.org/10.1145/2623330.2623623) |
| NS Survey | Negative sampling in knowledge graph representation learning: A review | arXiv | 2023 | [link](https://arxiv.org/abs/2301.12345) |
| Uniform Sampling | Knowledge graph embedding by translating on hyperplanes | AAAI | 2014 | [link](https://ojs.aaai.org/index.php/AAAI/article/view/8870) |
| KBGAN  | KBGAN: Adversarial learning for knowledge graph embeddings | NAACL | 2018 | [Link](https://doi.org/10.18653/v1/N18-1133) |
| Self-Adv | RotatE: Knowledge graph embedding by relational rotation in complex space | ICLR | 2019 | [link](https://arxiv.org/pdf/1902.10197) |
| Batch NS | Pytorch-BigGraph: A large scale graph embedding system | Machine Learning and Systems | 2019 | [link](https://proceedings.mlsys.org/paper_files/paper/2019/hash/1eb34d662b67a14e3511d0dfd78669be-Abstract.html) |
| Bernoulli NS | An interpretable knowledge transfer model for knowledge base completion | ACL | 2017 | [link](https://doi.org/10.18653/v1/P17-1088) |
| Zhang et al. | A novel negative sample generating method for knowledge graph embedding | EWSN | 2019 | [link](https://www.ewsn.org/file-repository/ewsn2019/401_406_zhang.pdf) |
| SparseNSG | A novel negative sampling based on frequency of relational association entities for knowledge graph embedding | Journal of Web Engineering | 2021 | [link](https://doi.org/10.13052/jwe1540-9589.2068) |
| IGAN | Incorporating GAN for negative sampling in knowledge representation learning | AAAI | 2018 | [link](https://ojs.aaai.org/index.php/AAAI/article/view/11536) |
| GraphGAN | GraphGAN: Graph representation learning with generative adversarial nets | AAAI | 2018 | [link](https://ojs.aaai.org/index.php/AAAI/article/view/11872) |
| KSGAN | A knowledge selective adversarial network for link prediction in knowledge graph | NLPCC | 2019 | [link](https://doi.org/10.1007/978-3-030-32233-5_14) |
| RUGA | Improving knowledge graph completion using soft rules and adversarial learning | Chinese Journal of Electronics | 2021 | [link](https://doi.org/10.1049/cje.2021.05.004) |
| LAS | Adversarial knowledge representation learning without external model | IEEE Access | 2019 | [link](https://doi.org/10.1109/ACCESS.2018.2889481) |
| ASA | Relation-aware graph attention model with adaptive self-adversarial training | AAAI | 2021 | [link](https://ojs.aaai.org/index.php/AAAI/article/view/17129) |
| AN | Knowledge graph embedding based on adaptive negative sampling | ICPSEE | 2019 | [link](https://doi.org/10.1007/978-981-15-0118-0_42) |
| EANS | Entity aware negative sampling with auxiliary loss of false negative prediction for knowledge graph embedding | arXiv | 2022 | [link](https://arxiv.org/abs/2210.06242) |
| Truncated NS | Fusing attribute character embeddings with truncated negative sampling for entity alignment | Electronics | 2023 | [link](https://doi.org/10.3390/electronics12081947) |
| DNS | Distributional negative sampling for knowledge base completion | arXiv | 2019 | [link](https://arxiv.org/abs/1908.06178) |
| ESNS | Entity similarity-based negative sampling for knowledge graph embedding | PRICAI | 2022 | [Link](https://doi.org/10.1007/978-3-031-20865-2_6) |
| RCWC | KGBoost: A classification-based knowledge base completion method with negative sampling | Pattern Recognition Letters | 2022 | [link](https://www.sciencedirect.com/science/article/abs/pii/S0167865522000939) |
| Conditional Sampling | Conditional constraints for knowledge graph embeddings | DL4KG | 2020 | [link](https://ceur-ws.org/Vol-2635/paper3.pdf) |
| LEMON | LEMON: LanguagE MOdel for negative sampling of knowledge graph embeddings | arXiv preprint | 2022 | [Link](https://arxiv.org/abs/2203.04703) |
| NSCaching | NSCaching: Simple and efficient negative sampling for knowledge graph embedding | ICDE | 2019 | [Link](https://doi.org/10.1109/ICDE.2019.00061) |
| MDNcaching | MDNcaching: A strategy to generate quality negatives for knowledge graph embedding | IEA/AIE | 2022 | [Link](https://doi.org/10.1007/978-3-031-08530-7_74) |
| Op-Trans | Op-Trans: An optimization framework for negative sampling and triplet-mapping properties in knowledge graph embedding | Applied Sciences | 2023 | [Link](https://doi.org/10.3390/app13052817) |
| NS-KGE | Efficient non-sampling knowledge graph embedding | The Web Conference | 2021 | [Link](https://doi.org/10.1145/3442381.3449859) |
