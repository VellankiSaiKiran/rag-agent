# Cosine similarity

Cosine similarity measures how similar two vectors are by the angle between them, not their length. A score of 1 means the vectors point the same way, 0 means they are unrelated, and negative values mean they point in opposite directions.

In retrieval, the query and each document chunk are turned into vectors, and the chunks with the highest cosine scores are returned first.