# Changelog

### [2.0.0] - 2025-10-02
- Added support for multiple white matter bundle formats: `trk`, `trx`, `tt.gz`, `tck`.  
- Added option to invert the transform matrix.  
- Fixed along-length segmentation/labeling issue when subject bundles were cropped or incomplete.  
- Added option to extract volumetric and streamline profiles.  
- Added streamline shape metrics.  

---

### [1.0.1] - 2025-02-21
- Fixed issue where `excluded_idx` was empty, preventing along-length segmentation.  
- Prevented creation of empty DataFrames when computing bundle features from binary masks and microstructure maps.  

---

### [1.0.0] - 2024-11-02
Initial release of **Medial Tractography Analysis (MeTA)**: a workflow for minimizing brain microstructural heterogeneity in diffusion MRI (dMRI) metrics by extracting and parcellating the core volume along the length of white matter bundles in voxel space.  

**Features:**  
- Extraction of the medial surface and core volume of white matter (WM) bundles.  
- Segmentation of WM bundles along their length in voxel space.  
- Computation of volumetric and streamline-based features.  
