# TabiBench Comparison Report


## Detailed Results

| Type                 | Task                               |   boun-tabilab/TabiBERT (36d24b) |   magibu/embeddingmagibu-200m (e1351a) |
|:---------------------|:-----------------------------------|---------------------------------:|---------------------------------------:|
| Classification       | BilTweetNews-Sentiment-Analysis    |                            35.87 |                                  42.07 |
| Classification       | Gender-Hate-Speech-TR              |                            43.05 |                                  39.91 |
| Classification       | News-Cat                           |                            84.60 |                                  88.76 |
| Classification       | Pubmed-RCT-10K-TR                  |                            46.45 |                                  44.46 |
| Classification       | Sci-Cite-TR                        |                            47.49 |                                  47.54 |
| Classification       | Thesis-Abstract-Classification-11K |                           nan    |                                  36.67 |
| Classification       | Turkish-Product-Reviews            |                            62.63 |                                  73.86 |
| Classification (NLI) | Med-NLI-TR                         |                            36.56 |                                  36.16 |
| Classification (NLI) | MultiNLI-TR                        |                            36.33 |                                  36.71 |
| Classification (NLI) | SNLI-TR                            |                            37.03 |                                  38.52 |
| Retrieval            | Apps-Retrieval-TR                  |                           nan    |                                   6.07 |
| Retrieval            | Code-Search-Net-21K-TR             |                           nan    |                                  73.67 |
| Retrieval            | Cos-QA-TR                          |                             2.16 |                                  51.80 |
| Retrieval            | Fiqa-TR                            |                             4.30 |                                  38.06 |
| Retrieval            | MsMarco-TR                         |                             8.17 |                                  79.15 |
| Retrieval            | NFCorpus-TR                        |                             0.07 |                                   1.16 |
| Retrieval            | Quora-TR                           |                            58.55 |                                  73.52 |
| Retrieval            | Scifact-TR                         |                            11.12 |                                  68.04 |
| Retrieval            | Stackoverflow-QA-TR                |                           nan    |                                  75.52 |
| STS                  | STSb-TR                            |                             0.00 |                                   0.00 |

## Average Scores by Type

| Type                 |   boun-tabilab/TabiBERT (36d24b) |   magibu/embeddingmagibu-200m (e1351a) |
|:---------------------|---------------------------------:|---------------------------------------:|
| Classification       |                            53.35 |                                  53.32 |
| Classification (NLI) |                            36.64 |                                  37.13 |
| Retrieval            |                            14.06 |                                  51.89 |
| STS                  |                             0.00 |                                   0.00 |