## Medial Tractography Analysis (MeTA)

<p align="center">
<img width="800" alt="workflow" src="https://github.com/bagari/meta/blob/main/resources/MeTA_workflow.png">
</p>

MeTA is a workflow implemented to minimize microstructural heterogeneity in diffusion MRI (dMRI) metrics by extracting and parcellating the core volume along the bundle length in the voxel-space directly while effectively preserving bundle shape and efficiently capturing the regional variation within and along white matter (WM) bundles.


If you use MeTA code, please cite the following publication:
* Heritability and Genetic Correlations Along the Corticospinal Tract. In International Workshop on Computational Diffusion MRI, CDMRI 2024 (Accepted)
* [Ba Gari, I., et al.: Medial tractography analysis (MeTA) for white matter population analyses across datasets. In: 2023 11th International IEEE/EMBS Conference on Neural Engineering (NER). pp. 1–5 (Apr 2023)](https://doi.org/10.1109/NER52421.2023.10123727)
* [Ba Gari, I., et al.: Along-tract parameterization of white matter microstructure using medial tractography analysis (MeTA). In: The 19th International Symposium on Medical Information Processing and Analysis (2023)](https://doi.org/10.1109/SIPAIM56729.2023.10373540)
* [Yushkevich, P.A.: Continuous medial representation of brain structures using the biharmonic PDE. Neuroimage 45(1 Suppl), S99–110 (Mar 2009)](https://doi.org/10.1016/j.neuroimage.2008.10.051)

## Installation
1) Clone this repository:
```
git clone https://github.com/USC-LoBeS/MeTA
```
2) Navigate to the meta folder and then create a virtual environment using 
```
conda env create -f environment.yml
```
3) Activate the environment:
```
conda activate meta
```
4) Install MeTA package
```
python -m pip install .
```
5) Build the CMREP C++ code:
```
sh build.sh
```
> NOTE: Use `meta --help` to see the package options.

## How to use the package:
* Convert streamlines in trk format to binary image
```
meta_bundle_density --bundle CST.trk --reference dti_FA.nii.gz --output CST.nii.gz
```

* Generating 3D Medial surface for WM bundle using CMREP method: 
```
vtklevelset CST.nii.gz CST.vtk 0.1
cmrep_vskel -c 3 -p 1.5 -g CST.vtk CST_skeleton.vtk
````

* Running Medial Tractography Analysis (MeTA):
```
meta --subject 1234 --bundle CST --medial_surface CST_skeleton.vtk --volume CST.vtk --sbundle CST.trk --mbundle CST_model.trk --mask CST.nii.gz --percent 0.1 --num_segments 15 --output CST
```

* Extracting segment features:
```
meta_segment_features --subject 1234 --bundle CST --mask CST_segments_local_core.nii.gz --map FA.nii.gz --output CST_FA_15_segments_local_core_metrics.csv
```

* Extracting streamline features:

```
meta_streamlines_features --subject 1234 --bundle CST --core CST_global_core.nii.gz --all CST.nii.gz --tractogram CST.trk --output CST_streamlines_metrics.csv
```

## Contact: 
Iyad Ba Gari <bagari@usc.edu>
