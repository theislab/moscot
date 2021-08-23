# scOTT
Single-Cell Optimal Transport Tools

scOTT is intended to be a general framework to apply tools from optimal transport to time-course single-cell data. It should support
- multimodal data (esp. ATAC & RNA)
- lineage-tracing data (procpective & retrospective)

while scaling to large (~10k cells per time point) samples. In the backend, it will be based on either [OTT](https://ott-jax.readthedocs.io/en/latest/index.html) of [GeomLoss](https://www.kernel-operations.io/geomloss/index.html) for fast and memory-efficient computations. 
