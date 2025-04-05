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

<div align="center">
  <h3><strong>Open-Source Library for KGE</strong></h3>
</div>

| Library | Implementation | Key Features | GitHub Repository |
|:---:|:---:|---|---|
| OpenKE | Pytorch, TensorFlow, C++ | Efficiently implements fundamental operations such as data loading, negative sampling, and performance evaluation using C++ for high performance. | [https://github.com/thunlp/OpenKE](https://github.com/thunlp/OpenKE) |
| AmpliGraph | TensorFlow | Provides a Keras-style API with improved efficiency over OpenKE. | [https://github.com/Accenture/AmpliGraph](https://github.com/Accenture/AmpliGraph) |
| torchKGE | Pytorch | Achieves twice the efficiency of OpenKE and five times that of AmpliGraph. | [https://github.com/torchkge-team/torchkge](https://github.com/torchkge-team/torchkge) |
| LibKGE | Pytorch | Enables direct configuration of hyperparameters and model settings via configuration files. | [https://github.com/uma-pi1/kge](https://github.com/uma-pi1/kge) |
| KB2E | C++ | One of the earliest KGE libraries and the predecessor of OpenKE. | [https://github.com/thunlp/KB2E](https://github.com/thunlp/KB2E) |
| scikit-kge | Python | Implements multiple classical KGE models and supports a novel negative sampling strategy. | [https://github.com/mnick/scikit-kge](https://github.com/mnick/scikit-kge) |
| NeuralKG | Pytorch | Integrates KGE techniques with graph neural networks (GNNs) and rule-based reasoning models. | [https://github.com/zjukg/NeuralKG](https://github.com/zjukg/NeuralKG) |
| PyKEEN | Pytorch | Offers 37 datasets, 40 KGE models, 15 loss functions, 6 regularization mechanisms, and 3 negative sampling strategies. | [https://github.com/pykeen/pykeen](https://github.com/pykeen/pykeen) |
| Pykg2vec | Pytorch, TensorFlow | Supports automated hyperparameter tuning, exports KG embeddings in TSV or RDF formats, and provides visualization for performance evaluation. | [https://github.com/Sujit-O/pykg2vec](https://github.com/Sujit-O/pykg2vec) |
| μKG | Pytorch, TensorFlow | Supports multi-process execution and GPU-accelerated computation, making it well-suited for large-scale KGs. | [https://github.com/nju-websoft/muKG](https://github.com/nju-websoft/muKG) |
| DGL-KE | Pytorch, MXNet | Optimized for execution on CPU and GPU clusters, offering high scalability for large-scale KGs. | [https://github.com/awslabs/dgl-ke](https://github.com/awslabs/dgl-ke) |
| GraphVite | Pytorch | Provides efficient large-scale embedding learning, supports visualization of graph data, and enables multi-processing and GPU parallelization. | [https://github.com/DeepGraphLearning/graphvite](https://github.com/DeepGraphLearning/graphvite) |
| PBG | Pytorch | Designed for distributed training, capable of handling KGs with billions of entities and trillions of edges. | [https://github.com/facebookresearch/PyTorch-BigGraph](https://github.com/facebookresearch/PyTorch-BigGraph) |

### Logic Rule-based KGR Model

<div align="center">
  <h3><strong>Rule Learning for KG</strong></h3>
</div>

| Model  | Title | Conference/Journal | Year | Paper |
|:-----:|---------------------------------|:---------------------------------:|:------:|:------:|
| FOIL | Learning logical definitions from relations | Machine Learning | 1990 | [link](https://doi.org/10.1023/A:1022699322624) |
| MDIE | Inverse entailment and progol | New Generation Computing | 1995 | [link](https://doi.org/10.1007/BF03037227) |
| Inspire | Best-effort inductive logic programming via fine-grained cost-based hypothesis generation | Machine Learning | 2018 | [link](https://doi.org/10.1007/s10994-018-5708-2) |
| Neural-Num-LP | Differentiable learning of numerical rules in knowledge graphs | ICLR | 2020 | [link](https://openreview.net/pdf?id=rJleKgrKwS) |
| AMIE+ | Fast rule mining in ontological knowledge bases with AMIE+ | VLDB Journal | 2015 | [link](https://doi.org/10.1007/s00778-015-0394-1) |
| ScaLeKB | ScaLeKB: Scalable learning and inference over large knowledge bases | VLDB Journal | 2016 | [link](https://doi.org/10.1007/s00778-016-0444-3) |
| RDF2rules | RDF2Rules: Learning rules from RDF knowledge bases by mining frequent predicate cycles | arXiv | 2015 | [link](https://arxiv.org/abs/1512.07734) |
| SWARM | SWARM: An approach for mining semantic association rules from semantic web data | PRICAI | 2016 | [link](https://doi.org/10.1007/978-3-319-42911-3_3) |
| Rudik | Rudik: Rule discovery in knowledge bases | PVLDB | 2018 | [link](https://doi.org/10.14778/3229863.3236231) |
| RuLES | Rule learning from knowledge graphs guided by embedding models | ESWC | 2018 | [link](https://doi.org/10.1007/978-3-030-00671-6_5) |
| Evoda | Rule learning over knowledge graphs with genetic logic programming | ICDE | 2022 | [link](https://doi.org/10.1109/ICDE53745.2022.00318) |
| NeuralLP | Differentiable learning of logical rules for knowledge base reasoning | NeurIPS | 2017 | [link](https://proceedings.neurips.cc/paper/2017/file/0e55666a4ad822e0e34299df3591d979-Paper.pdf) |
| DRUM | DRUM: End-to-end differentiable rule mining on knowledge graphs | NeurIPS | 2019 | [link](https://proceedings.neurips.cc/paper_files/paper/2019/file/0c72cb7ee1512f800abe27823a792d03-Paper.pdf) |
| RLvLR | An embedding-based approach to rule learning in knowledge graphs | IEEE TKDE | 2019 | [link](https://doi.org/10.1109/TKDE.2019.2941685) |
| RNNLogic | RNNLogic: learning logic rules for reasoning on knowledge graphs | ICLR | 2021 | [link](https://arxiv.org/pdf/2010.04029) |
| RARL | Relatedness and TBox-driven rule learning in large knowledge bases | AAAI | 2020 | [link](https://ojs.aaai.org/index.php/AAAI/article/view/5690/) |
| Ruleformer | Ruleformer: context-aware rule mining over knowledge graph | COLING | 2022 | [link](https://aclanthology.org/2022.coling-1.225/) |
| Ott et al. | Rule-based knowledge graph completion with canonical models | CIKM | 2023 | [link](https://doi.org/10.1145/3583780.3615042) |

<div align="center">
  <h3><strong>Neural-Symbolic KGR</strong></h3>
</div>

| Model  | Title | Conference/Journal | Year | Paper |
|:-----:|---------------------------------|:---------------------------------:|:------:|:------:|
| KALE | Jointly embedding knowledge graphs and logical rules | EMNLP | 2016 | [link](https://doi.org/10.18653/v1/D16-1019) |
| RUGE | Knowledge graph embedding with iterative guidance from soft rules | AAAI | 2018 | [link](https://ojs.aaai.org/index.php/AAAI/article/view/11918) |
| RulE | RulE: Knowledge graph reasoning with rule embedding | Findings of ACL | 2024 | [link](https://doi.org/10.18653/v1/2024.findings-acl.256) |
| RPJE | Rule-guided compositional representation learning on knowledge graphs | AAAI | 2020 | [link](https://doi.org/10.1609/aaai.v34i03.5687) |
| IterE | Iteratively learning embeddings and rules for knowledge graph reasoning | WWW | 2019 | [link](https://doi.org/10.1145/3308558.3313612) |
| UniKER | UniKER: A unified framework for combining embedding and definite Horn rule reasoning for knowledge graph inference | EMNLP | 2021 | [link](https://doi.org/10.18653/v1/2021.emnlp-main.769) |
| EngineKG | Perform like an engine: A closed-loop neural-symbolic learning framework for knowledge graph inference | COLING | 2022 | [link](https://aclanthology.org/2022.coling-1.119/) |

## Static Multi-Step KGR

### Random Walk-based Models

<div align="center">
  <h3><strong>Random Walk-based Models</strong></h3>
</div>

| Model  | Title | Conference/Journal | Year | Paper |
|:-----:|---------------------------------|:---------------------------------:|:------:|:------:|
| PRA | Relational retrieval using a combination of path-constrained random walks | *Machine Learning* | 2010 | [link](https://doi.org/10.1007/s10994-010-5205-8) |
| Lao et al. 1 | Random walk inference and learning in a large scale knowledge base | EMNLP | 2011 | [link](https://aclanthology.org/D11-1049/) |
| Lao et al. 2 | Reading the web with learned syntactic-semantic inference rules | EMNLP | 2012 | [link](https://aclanthology.org/D12-1093/) |
| Gardner et al. | Improving learning and inference in a large knowledge-base using latent syntactic cues | EMNLP | 2013 | [link](https://doi.org/10.18653/v1/D15-1173) |
| CPRA | Knowledge base completion via coupled path ranking | ACL | 2016 | [link](https://doi.org/10.18653/v1/P16-1124) |
| C-PR | Context-aware path ranking for knowledge base completion | IJCAI | 2017 | [link](https://www.ijcai.org/Proceedings/2017/166) |
| A\*Net | A\*Net: a scalable path-based reasoning approach for knowledge graphs | NeurIPS | 2024 | [link](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b9e98316cb72fee82cc1160da5810abc-Abstract-Conference.html) |
| SFE | Efficient and expressive knowledge base completion using subgraph feature extraction | EMNLP | 2015 | [link](https://doi.org/10.18653/v1/D15-1173) |
| PathCon | Relational message passing for knowledge graph completion | KDD | 2021 | [link](https://doi.org/10.1145/3447548.3467247) |

<div align="center">
  <h3><strong>Reinforcement Learning-based Models</strong></h3>
</div>

| Model  | Title | Conference/Journal | Year | Paper |
|:-----:|---------------------------------|:---------------------------------:|:------:|:------:|
| DeepPath   | DeepPath: a reinforcement learning method for knowledge graph reasoning | EMNLP | 2017 | [link](https://doi.org/10.18653/v1/D17-1060) |
| MINERVA | Go for a walk and arrive at the answer: Reasoning over paths in knowledge bases using reinforcement learning | ICLR | 2018 | [link](https://www.akbc.ws/2017/papers/24_paper.pdf) |
| DIVA  | Variational knowledge graph reasoning | NAACL | 2018 | [link](https://doi.org/10.18653/v1/N18-1165) |
| MultiHopKG  | Multi-hop knowledge graph reasoning with reward shaping | EMNLP | 2018 | [link](https://doi.org/10.18653/v1/D18-1362) |
| M-Walk  | M-Walk: Learning to walk over graphs using monte carlo tree search | NeurIPS | 2018 | [link]() |
| RARL  | Rule-aware reinforcement learning for knowledge graph reasoning | ACL-IJCNLP | 2021 | [link](https://doi.org/10.18653/v1/2021.findings-acl.412) |
| AttnPath  | Incorporating graph attention mechanism into knowledge graph reasoning based on deep reinforcement learning | EMNLP-IJCNLP | 2019 | [link](https://doi.org/10.18653/v1/D19-1264) |
| DIVINE  | DIVINE: A generative adversarial imitation learning framework for knowledge graph reasoning | EMNLP-IJCNLP | 2019 | [link](https://doi.org/10.18653/v1/D19-1266) |

<div align="center">
  <h3><strong>LLM-based Multi-Step KGR Models</strong></h3>
</div>

| Model  | Title | Conference/Journal | Year | Paper |
|:-----:|---------------------------------|:---------------------------------:|:------:|:------:|
| KG\&LLM Survey | Unifying large language models and knowledge graphs: A roadmap | IEEE TKDE | 2024 | [link](https://doi.org/10.1109/TKDE.2024.3352100) |
| StructGPT | StructGPT: A general framework for large language model to reason over structured data | EMNLP | 2023 | [link](https://doi.org/10.18653/v1/2023.emnlp-main.574) |
| KSL | Knowledge solver: Teaching LLMs to search for domain knowledge from knowledge graphs | arXiv | 2023 | [link](https://arxiv.org/abs/2309.03118) |
| KD-CoT | Knowledge-driven CoT: Exploring faithful reasoning in LLMs for knowledge-intensive question answering | arXiv | 2023 | [link](https://arxiv.org/abs/2308.13259) |
| ToG | Think-on-Graph: Deep and responsible reasoning of large language model on knowledge graph | ICLR | 2024 | [link](https://arxiv.org/pdf/2307.07697) |
| KnowledgeNavigator | KnowledgeNavigator: Leveraging large language models for enhanced reasoning over knowledge graph | Complex Intell. Syst. | 2024 | [link](https://doi.org/10.1007/s40747-024-01527-8) |
| Nguyen et al. | Direct evaluation of chain-of-thought in multi-hop reasoning with knowledge graphs | Findings of ACL | 2024 | [link](https://doi.org/10.18653/v1/2024.findings-acl.168) |
| KG-Agent | KG-Agent: An efficient autonomous agent framework for complex reasoning over knowledge graph | arXiv | 2024 | [link](https://arxiv.org/abs/2402.11163) |
| AgentTuning | AgentTuning: Enabling generalized agent abilities for LLMs | Findings of ACL | 2024 | [link](https://doi.org/10.18653/v1/2024.findings-acl.181) |
| Glam | Glam: Fine-tuning large language models for domain knowledge graph alignment via neighborhood partitioning and generative subgraph encoding | AAAI Symposium | 2024 | [link](https://ojs.aaai.org/index.php/AAAI-SS/article/view/31186) |

## Dynamic KGR
### Incremental KGE Model

<div align="center">
  <h3><strong>Incremental KGE Model</strong></h3>
</div>

| Model  | Title | Conference/Journal | Year | Paper |
|:-----:|---------------------------------|:---------------------------------:|:------:|:------:|
| DKGE | Efficiently embedding dynamic knowledge graphs | Knowl.-Based Syst. | 2022 | [link](https://doi.org/10.1016/j.knosys.2022.109124) |
| PuTransE | Non-parametric estimation of multiple embeddings for link prediction on dynamic knowledge graphs | AAAI | 2017 | [link](https://doi.org/10.1609/aaai.v31i1.10685) |
| Liu et al. | Heuristic-driven, type-specific embedding in parallel spaces for enhancing knowledge graph reasoning | ICASSP | 2024 | [link]() |
| ABIE | Anchors-based incremental embedding for growing knowledge graphs | TKDE | 2023 | [link](https://doi.org/10.1109/TKDE.2021.3136482) |
| CKGE | Towards continual knowledge graph embedding via incremental distillation | AAAI | 2024 | [link](https://doi.org/10.1609/aaai.v38i8.28722) |
| LKGE | Lifelong embedding learning and transfer for growing knowledge graphs | AAAI | 2023 | [link](https://doi.org/10.1609/aaai.v37i4.25539) |
| AIR | AIR: Adaptive incremental embedding updating for dynamic knowledge graphs | DASFAA | 2023 | [link](https://doi.org/10.1007/978-3-031-30672-3_41) |
| TIE | TIE: A framework for embedding-based incremental temporal knowledge graph completion | SIGIR | 2021 | [link](https://doi.org/10.1145/3404835.3462961) |
| RotatH | Incremental update of knowledge graph embedding by rotating on hyperplanes | ICWS | 2021 | [link](https://doi.org/10.1109/ICWS53863.2021.00072) |
| MMRotatH | Knowledge graph incremental embedding for unseen modalities | Knowl. Inf. Syst. | 2023 | [link](https://doi.org/10.1007/s10115-023-01868-9) |
| DKGE | Efficiently embedding dynamic knowledge graphs | Knowl.-Based Syst. | 2022 | [link](https://doi.org/10.1016/j.knosys.2022.109124) |
| Navi | Dynamic knowledge graph embeddings via local embedding reconstructions | ESWC (Satellite) | 2022 | [link](https://doi.org/10.1007/978-3-031-11609-4_36) |
| UOKE | Online updates of knowledge graph embedding | Complex Networks X | 2021 | [link](https://doi.org/10.1007/978-3-030-93413-2_44) |
| 257 | Temporal knowledge graph incremental construction model for recommendation | APWeb-WAIM | 2020 | [link](https://doi.org/10.1007/978-3-030-60259-8_26) |

### Temporal KGR Model

<div align="center">
  <h3><strong>Time Embedding-based Models</strong></h3>
</div>

| Model  | Title | Conference/Journal | Year | Paper |
|:-----:|---------------------------------|:---------------------------------:|:------:|:------:|
| TA-TransE   | Learning sequence encoders for temporal knowledge graph completion | EMNLP | 2018 | [link](https://doi.org/10.18653/v1/D18-1516) |
| HyTE   | HyTE: Hyperplane-based temporally aware knowledge graph embedding | EMNLP | 2018 | [link](https://doi.org/10.18653/v1/D18-1225) |
| TTransE  | Deriving validity time in knowledge graph | WWW | 2018 | N/A |
| TERO  | TeRo: A time-aware knowledge graph embedding via temporal rotation | COLING | 2020 | [link](https://doi.org/10.18653/v1/2020.coling-main.139) |
| TDistMult  | Embedding models for episodic knowledge graphs | JWS | 2019 | [link](https://doi.org/10.1016/j.websem.2018.12.008) |
| TComplEx  | Tensor decompositions for temporal knowledge base completion | ICLR | 2020 | N/A |
| SimplE  | Diachronic embedding for temporal knowledge graph completion | AAAI | 2020 | N/A |
| ATiSE  | Temporal KGC based on time series gaussian embedding | ISWC | 2020 | [link](https://doi.org/10.1007/978-3-030-62419-4_37) |
| TARGAT  | TARGAT: A time-aware relational graph attention model | IEEE/ACM TASLP | 2023 | [link](https://doi.org/10.1109/TASLP.2023.3282101) |
| LCGE  | Logic and commonsense-guided TKGC | AAAI | 2023 | [link](https://doi.org/10.1609/aaai.v37i4.25579) |

<div align="center">
  <h3><strong>Evolution Learning-based Models</strong></h3>
</div>

| Model  | Title | Conference/Journal | Year | Paper |
|:-----:|---------------------------------|:---------------------------------:|:------:|:------:|
| Know-Evolve  | Know-evolve: deep temporal reasoning for dynamic knowledge graphs | ICML  | 2017 | - |
| RE-NET  | Recurrent event network: autoregressive structure inference over temporal knowledge graphs | EMNLP | 2020 | - |
| CyGNet  | Learning from history: modeling temporal knowledge graphs with sequential copy-generation networks | AAAI  | 2021 | - |
| CluSTeR  | Search from history and reason for future: two-stage reasoning on temporal knowledge graphs | ACL   | 2021 | - |

<div align="center">
  <h3><strong>Temporal Rule Learning</strong></h3>
</div>

| Model  | Title | Conference/Journal | Year | Paper |
|:-----:|---------------------------------|:---------------------------------:|:------:|:------:|
| StreamLearner | Learning temporal rules from knowledge graph streams | AAAI Spring Symposium | 2019 | - |
| Tlogic | Tlogic: temporal logical rules for explainable link forecasting on temporal knowledge graphs | AAAI | 2022 | [link](https://doi.org/10.1609/aaai.v36i4.20330) |
| TILP | TILP: differentiable learning of temporal logical rules on knowledge graphs | ICLR | 2023 | - |
| TEILP | TEILP: time prediction over knowledge graphs via logical reasoning | AAAI | 2024 | - |
| NeuSTIP | NeuSTIP: a neuro-symbolic model for link and time prediction in temporal knowledge graphs | EMNLP | 2023 | [link](https://doi.org/10.18653/v1/2023.emnlp-main.274) |
