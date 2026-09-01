# v1.0 release checklist

## Metadata to replace

- [ ] Replace all `REPLACE_...` text in `README.md`, `CITATION.cff`, and `Data/README.md`.
- [ ] Replace `JPDFusion authors` in `LICENSE` with the copyright holder(s).
- [ ] Add the final paper DOI and public-data DOI.
- [ ] Confirm redistribution terms for every example GeoTIFF.

Find remaining placeholders:

```bash
git grep -n "REPLACE_\|10.XXXX"
```

## Validate the exact release tree

```bash
conda env create -f environment.yml
conda activate jpdfusion
python -m pip install -e .
python run_all.py --config config.example.yml --steps validate
python scripts/create_data_manifest.py --config config.example.yml
python -m pytest
git status
```

Run the complete example and compare its final products with the archived expected outputs before tagging.

## Commit and tag

```bash
git init
git lfs install
git add .
git commit -m "Release JPDFusion v1.0"
git branch -M main
git remote add origin https://github.com/REPLACE_WITH_OWNER/JPDFusion.git
git push -u origin main
git tag -a v1.0.0 -m "JPDFusion reproducibility release v1.0.0"
git push origin v1.0.0
```

On GitHub, create a release from tag `v1.0.0`, use the title **JPDFusion v1.0.0**, include the paper/data DOI, and attach a source archive if desired. Archive the GitHub release in Zenodo or another DOI-issuing service if a citable software DOI is required.

## Suggested release notes

> First public reproducibility release of JPDFusion. Includes single-band HLS example support; SAR and MCD43A4 auxiliary preprocessing; temporally aligned clustering; cluster-wise Gaussian-Copula conditional prediction; residual compensation; YAML configuration; input/output manifests; tests; and documentation. Real example data and their checksums are identified by the accompanying dataset DOI.

