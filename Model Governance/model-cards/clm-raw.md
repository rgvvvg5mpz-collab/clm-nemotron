# Model card · `clm-raw` (no trained head)

The encoder's own cosine similarity between the state and each label description, times a
fixed scale of 100, then softmax. Always available; the default model when no head is loaded.

- **Use for:** checking that the service works, ranking candidates, comparing label wording.
- **Do not use for:** decisions, thresholds or reported confidence. It scored 37.8% on the
  typed-decisions test, below the 45.0% from always guessing the most common label. It
  reached 56–57% on the five-topic support test, against 94–95% with a trained head.
- **Why:** the encoder was trained for retrieval and rewards word overlap. Its probabilities
  are extreme because a cosine gap of 0.05 times 100 becomes a 99% to 1% split.
