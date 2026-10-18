# New model checklist

Copy this file next to the run folder, fill it in, and keep it with the model card. A
head may serve real decisions only when every box is ticked by someone other than its trainer.

**Model:** `__________`  **Run folder:** `__________`  **Trainer:** ______  **Validator:** ______  **Date:** ______

## Purpose and scope
- [ ] The decision it supports, who acts on its output, and what happens when it is wrong are written down.
- [ ] It is not used for decisions with legal or similarly significant effects on people without human review (doc 05, "Not suitable for").

## Sourcing
- [ ] Encoder is listed in `src/clm/encoders.py`, and the head's checkpoint records it.
- [ ] Origin and lineage of every model involved were checked against the makers' model cards (doc 06). Lineage notes are added to doc 03.
- [ ] Licences allow this use, and notices are kept.

## Data
- [ ] We have the right to use the training data. Its source, collection date and labelling process are recorded in doc 03.
- [ ] Labels were produced or reviewed by people for this purpose (not only by a model).
- [ ] Every label has at least 30 examples, or the shortfall is documented.
- [ ] Test and check splits are held out by customer, account or case (`--group-column`), not by row.
- [ ] Personal data has a retention and deletion plan for `runs/`.

## Evaluation
- [ ] Test accuracy clearly beats both the majority label and `clm-raw` (by more than the noise band in doc 04).
- [ ] Per-label (or per-question) accuracy is reviewed; weak labels are noted.
- [ ] Calibration is fitted (temperature) and measured (ECE or Brier) on validation data; thresholds are set from validation.
- [ ] Spot-checked on at least 50 recent real examples labelled by people.
- [ ] Latency is measured on the target hardware.
- [ ] Fairness is assessed where outputs affect people.

## Release
- [ ] `questions.json` (exact question and label descriptions) is versioned with the head.
- [ ] Model card written, inventory (doc 02) and change log (doc 08) updated.
- [ ] Shipped in a new `models-vN` release with checksums; rollback release identified.
- [ ] Monitoring sample and retraining triggers agreed (doc 07).
