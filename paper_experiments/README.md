# Paper Experiments

This directory is a clean, paper-focused reorganization of the main experiments in this project.

## Included Experiments

1. `01_Image_NewDataset_Experiment`
   - Source: `comparing_to_yoel_and_eitan/experiment_2026-03-05`
   - Purpose: image-space/Fourier-Bessel experiments on the new and old datasets, bandwidth search, and figure generation.

2. `02_PointCloud_SelfTuning_Experiment`
   - Source: `full_pointcloud_experiment`
   - Purpose: point-cloud embedding experiments, including self-tuning methods and bandwidth utilities.

3. `03_Double_Rotation_Parameterization`
   - Source: `double_rotation_parameterization_experiment`
   - Purpose: double-rotation parameterization pipeline and visualization flow.

4. `04_Final_SO3_SO2_Exp3`
   - Source: `exp3/successful_exp3`
   - Purpose: final SO(3)/SO(2) experiment scripts and supporting documentation.

## Important Notes

- Large generated pickle files and raw dataset bundles were intentionally **not copied** into this paper package.
- Lightweight summaries and publication figures are included where available.
- Use the copied scripts to generate fresh outputs from code and algorithm settings.
- This keeps the paper package lighter and reproducible.

## Reproducible Run Examples

### 1) Image Experiment (new dataset)

```bash
cd paper_experiments
python3 "01_Image_NewDataset_Experiment/run_experiment.py"
```

Then generate figures:

```bash
python3 "01_Image_NewDataset_Experiment/generate_all_figures.py"
```

### 2) Point-Cloud Experiment (self-tuning)

```bash
python3 "02_PointCloud_SelfTuning_Experiment/run_experiment.py" \
  --data-path "../data/data_3D.pkl" \
  --save-folder "outputs/pointcloud" \
  --save-name paper_run \
  --num-points 200 \
  --invariant-method invariant_features_self_tuning \
  --movement rotation \
  --bandwidth 1 \
  --noise 0 \
  --noise-tag SNR \
  --laplacian-type RWGL \
  --knn-k 30 \
  --render-pdf true
```

### 3) Double Rotation Parameterization

```bash
python3 "03_Double_Rotation_Parameterization/run_experiment.py" \
  --input-path "/path/to/roy_lederman_data/data" \
  --n-images 5000 \
  --output-dir "outputs/double_rotation_paper" \
  --t 10 \
  --num-neighbors 20 \
  --num-rotations 300
```

### 4) Final SO(3)/SO(2) (exp3 successful)

```bash
bash "04_Final_SO3_SO2_Exp3/run_experiment.sh"
```

## Suggested Paper Workflow

1. Re-run each experiment from this directory.
2. Save outputs under a dedicated paper output root (for example, `outputs/paper_*`).
3. Use the generated figures and logs for final comparison and reporting.
