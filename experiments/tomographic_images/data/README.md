# Data

Put the projection dataset here (or pass `--data-dir`):

- `projections-synth.pt`: the tomographic projection images, a torch tensor
  (or list of tensors) of shape `(num_images, L, L)`, from the trajectory in
  which the ψ-torsion angle of the 19th residue of Glucagon moves along a
  half-circle (made with ASPIRE from PDB 1GCN);
- `output_main.csv`: columns `frame` and `angle`, the ground-truth torsion
  angle of each frame in degrees, used to colour the figures.

Figure 5 uses the first 1000 images. The data are not in the repository; ask
the authors for them.
