# HNSW index

HNSW stands for Hierarchical Navigable Small World. It is a way of organizing a large collection of vectors so you can find the ones closest to a query quickly, without comparing the query against every single vector. It is one of the most widely used indexes in vector databases.

The idea is to connect each vector to a handful of its near neighbors, forming a graph. To search, you start at some entry point and keep hopping to whichever neighbor is closer to the query, until no neighbor is any closer. Because each vector links to both nearby and somewhat distant points, a search can cross the whole collection in a small number of hops.

The "hierarchical" part means there are several layers of this graph. The top layer holds only a few vectors with long-range links, and each layer below holds more vectors with shorter links. A search starts at the top to get into the right region fast, then drops down layer by layer to refine the result, a bit like zooming in on a map.

HNSW is approximate: it usually finds the true nearest neighbors, but not always. Settings control the trade-off. Giving each vector more links, or exploring more candidates during a search, improves accuracy but uses more memory and time. For most retrieval tasks, a small loss in accuracy is well worth the large gain in speed.
