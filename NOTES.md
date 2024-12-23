pitfalls:

- Looking for spiculated nodules or larger nodules
- Thresholds, not looking for <6mm.
- Soft tissue density, mass density, calification. Dense as bone is benign
- Spiculated border is significant.
- Score of how worrisome.
- Steps: picture of chest xray, cut out center part, stop at diaphram, two dark arches. Take out bone density, subtract out ribs. >1 cm.

- Day 1
  - Got a simple thing up and running from my template, put in dataset and stuff.
  - Inefficient data handling because of spooky pytorch type errors. Saving to a cache instead.
  - Simple CNN with ReLU for now.
  - Used BCELoss, other small tweaks
  - Letting it run, looks good! Acc is high.
  - Acc 0.82, put residuals in the code for later.
  - This is `runs/test1`.
  - Stopped early because plateau. Saved best val acc, 0.8.
  - `runs/v2-res` residual.
    - Weirdly, residual made it worse.
  - I would consider this a relative success, of course it can be finetuned, but overall cool to get a model up and running in a short amount of time.
