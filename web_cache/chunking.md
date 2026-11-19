# Chunking

Chunking is the step where long documents are split into smaller pieces before they are embedded and stored for retrieval. A model can only read so much text at once, and a single vector for a whole document blurs many topics together. Smaller pieces let the system pull out just the part that answers a question.

Chunk size is a balancing act. Chunks that are too small lose context: a sentence on its own may not make sense without the paragraph around it. Chunks that are too large mix several ideas, so their vectors match queries less precisely, and they waste space in the prompt. A few hundred words per chunk is a common starting point.

Where you cut matters as much as how big the pieces are. Splitting in the middle of a sentence or a table makes chunks harder to understand. Better approaches cut at natural boundaries such as headings, paragraphs, or sentences, and only fall back to a hard character limit when a section is too long.

Many systems also add overlap, repeating a little text from the end of one chunk at the start of the next. This helps when an important idea sits right on a boundary. It is also useful to keep each chunk's source and position with it, so answers can cite where the information came from.
