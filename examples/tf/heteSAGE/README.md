# heteSAGE

heteSAGE is semi-supervised graphSAGE model that support metapaths of heterogeneous features.

1. Heterogeneous features include sparse and dense features of multi type of vertexes.
1. Share weights between multi-metapaths
1. Use the attention mechanism to aggregate the results of multi-metapaths embedding of the same node type
1. Use graphSAGE aggregate method to aggregate multi-hops features

trainning sample contains <target id, context id, ..., label>
