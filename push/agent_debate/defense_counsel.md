# The Case for Publishing at ICONS 2026

making the strongest honest case for each contribution. this is the "yes we should submit" side of the argument.

---

## framing note

ICONS is not NeurIPS. its CFP explicitly welcomes "benchmark tasks for neuromorphic computing," "hardware deployment," and "algorithms and training" work. the 2022 most directly comparable paper (Yarga et al.) benchmarked 4 encodings on speech digits and was accepted. the 2025 best paper was about turbulence modeling using neuron random walks -- no classification accuracy at all.

the question isn't whether this is publishable at a top ML venue (it isn't). it's whether it's publishable at ICONS, which serves exactly the community this work is for.

---

## C1: First Convolutional SNN on ESC-50

not contested by any paper in existence. Larroza et al. explicitly state "no state-of-the-art solution has yet encoded environmental sound datasets using spike-based methods" and themselves only evaluate ESC-10 (10 classes), FC-only architecture, 3 encodings, no hardware. Basu et al. (2025) 24-page survey reaches same conclusion.

the novelty of a "first" is only as good as the benchmark itself. ESC-50 IS meaningful -- standard since Piczak 2015, human performance 81.3%, ANN SOTA 99.1%, predefined 5-fold CV for reproducibility. establishing the first SNN results creates the reference point the field literally cannot do without.

the matched-architecture comparison (SNN vs ANN under identical training) is methodologically clean: the 16.7pp gap is attributable to spiking mechanism alone, not architectural differences.

---

## C2: 7-Encoding Comparison

the most comprehensive encoding comparison ever conducted on a standard audio benchmark. the ordering (direct >> rate ~ phase > population > latency >> delta ~ burst) is internally consistent and mechanistically explicable. each failure has a different root cause: delta fails because static spectrograms have no temporal variation; burst fails because front-loading creates temporal window mismatch; latency fails because spectrogram features aren't naturally compatible with first-spike timing; population underperforms because MSE count loss is harder to optimize than CE rate loss.

