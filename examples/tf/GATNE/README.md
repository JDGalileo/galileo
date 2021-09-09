# Representation Learning for Attributed Multiplex Heterogeneous Network (GATNE)

- Paper link: [https://arxiv.org/abs/1905.01669](https://arxiv.org/abs/1905.01669)
- Author's code repo: [https://github.com/THUDM/GATNE](https://github.com/THUDM/GATNE).


## Dataset

These datasets are sampled from the original datasets.

- Amazon contains 10,166 nodes and 148,865 edges. [Source](http://jmcauley.ucsd.edu/data/amazon)
- Twitter contains 10,000 nodes and 331,899 edges. [Source](https://snap.stanford.edu/data/higgs-twitter.html)
- YouTube contains 2,000 nodes and 1,310,617 edges. [Source](http://socialcomputing.asu.edu/datasets/YouTube)
- Example contains 6,163 nodes and 17,865 edges.

We Download all datasets from [official data](https://github.com/THUDM/GATNE/tree/master/data)

## Training

To train transductive models on dataset(available ds-name: example, youtube, amazon,twitter)

```
python examples/tf/GATNE/transductive.py --ds-name example
```


## Results
All the results (25 epochs) match the [official code](https://github.com/THUDM/GATNE) with the same hyper parameter values.


|         |GATNE-T|GATNE-T|GATNE-I|GATNE-I|
| ------- | -----  | ----- | -----  | ----- |
|         | auc   |  f1   |  auc  |   f1  |
| amazon  | 95.02 | 89.70 | 84.20 | 77.99 |
| youtube | 81.45 | 74.09 | - | - |
| twitter |  |  | - | - |
| example | 93.29 | 87.38 | 91.57 | 85.22 |

We train GATNE-T on amazon for 25 epochs, on example for 10 epochs, on youtube for 5 epochs, and train GATNE-I on amazon for 5 epochs, on example for 20 epochs.