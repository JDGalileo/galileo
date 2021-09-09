GraphSAGE:  Inductive Representation Learning on Large Graphs

paper link: https://arxiv.org/abs/1706.02216

## Unsupervised GraphSage
Parameters:

| Name | Value |
| ---- | ----- |
|embedding_dim|64|
|batch_size|32|
|num_epochs|10|
|learning_rate|0.01|
|optimizer|adam|
|fanouts|10,10|
|aggregator|meanpool|
|num_negs|5|


## Supervised GraphSage
Parameters:

| Name | Value |
| ---- | ----- |
|embedding_dim|64|
|batch_size|32|
|num_epochs|10|
|learning_rate|0.01|
|optimizer|adam|
|fanouts|10,10|
|aggregator|meanpool|

