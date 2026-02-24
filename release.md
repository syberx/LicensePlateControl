# Packages included in the release

- intel-gmmlib (https://github.com/intel/gmmlib)
- intel-opencl-icd, intel-level-zero-gpu (https://github.com/intel/compute-runtime)

## Components revisions included in the release

- [intel/compute-runtime@24.13.29138.7](https://github.com/intel/compute-runtime/releases/tag/24.13.29138.7)
- [intel/gmmlib@intel-gmmlib-22.3.18](https://github.com/intel/gmmlib/releases/tag/intel-gmmlib-22.3.18)

## Additional components revisions used in build
- Used for building runtime
   - intel/libva@2.2.0 (Compatible with va_api_major_version = 1)
   - [oneapi-src/level-zero@v1.16.14](https://github.com/oneapi-src/level-zero/releases/tag/v1.16.14) (Supports [oneAPI Level Zero Specification v1.7.8](https://spec.oneapi.io/level-zero/1.7.8/index.html))
   - [intel/intel-graphics-compiler@igc-1.0.16510.2](https://github.com/intel/intel-graphics-compiler/releases/tag/igc-1.0.16510.2)
   - [intel/igsc@V0.8.16](https://github.com/intel/igsc/releases/tag/V0.8.16)

## Installation procedure on Ubuntu 22.04

1. Create temporary directory

Example:

```
mkdir neo
```

2. Download all *.deb packages

Example:

```
cd neo
wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.16510.2/intel-igc-core_1.0.16510.2_amd64.deb
wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.16510.2/intel-igc-opencl_1.0.16510.2_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/24.13.29138.7/intel-level-zero-gpu-dbgsym_1.3.29138.7_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/24.13.29138.7/intel-level-zero-gpu_1.3.29138.7_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/24.13.29138.7/intel-opencl-icd-dbgsym_24.13.29138.7_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/24.13.29138.7/intel-opencl-icd_24.13.29138.7_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/24.13.29138.7/libigdgmm12_22.3.18_amd64.deb
```

3. Verify sha256 sums for packages

Example:

```
wget https://github.com/intel/compute-runtime/releases/download/24.13.29138.7/ww13.sum
sha256sum -c ww13.sum
```
4. Install all packages as root

Example:

```
sudo dpkg -i *.deb
```
In case of installation problems, please install required dependencies, for example:
```
apt install ocl-icd-libopencl1
```

## sha256 sums for packages

```
7b3cf621e1a2740d24f6f5bcd8ec27a3c920cdfab7341c7b0e781693f112d64a  intel-igc-core_1.0.16510.2_amd64.deb
f0f84804beb063d1bd92ed2f9213540b53561fdf8027f9f57dc6fad6739a49d5  intel-igc-opencl_1.0.16510.2_amd64.deb
210c698a10ac6606720b058084089bb4579b5f2cc77f933ae8c1f82cc0da48a2  intel-level-zero-gpu_1.3.29138.7_amd64.deb
d414819a77554d751c3e73d00dae61119abdc37360514eedc6d7783b4185a871  intel-level-zero-gpu-dbgsym_1.3.29138.7_amd64.ddeb
2a4639f48d55f562d37f36af0baca42e78b833323429d2526871d5e75a197f63  intel-opencl-icd_24.13.29138.7_amd64.deb
d528e2bd7f1ca437e19a705b81b3badec88a569aa1fa61d3a1fbf29670e78eac  intel-opencl-icd-dbgsym_24.13.29138.7_amd64.ddeb
e952d8aeccefdb46f2c037524cbb31ff164c197255e9762c35326c0e83b9f582  libigdgmm12_22.3.18_amd64.deb
```

## Quality expectations

Platform | Quality | OpenCL | Level Zero | WSL
----------- | --- | ---| ---| -----------
[DG1](https://ark.intel.com/content/www/us/en/ark/products/codename/195485/products-formerly-dg1.html) | Production | 3.0 | 1.3 | Yes
[DG2](https://ark.intel.com/content/www/us/en/ark/products/codename/226095/products-formerly-alchemist.html) | Production | 3.0 | 1.3 | Yes
[Skylake](https://ark.intel.com/content/www/us/en/ark/products/codename/37572/skylake.html) | Production | 3.0 | 1.3 | --
[Kaby Lake](https://ark.intel.com/content/www/us/en/ark/products/codename/82879/kaby-lake.html) | Production | 3.0 | 1.3 | --
[Coffee Lake](https://ark.intel.com/content/www/us/en/ark/products/codename/97787/coffee-lake.html) | Production | 3.0 | 1.3 | Yes
[Ice Lake](https://ark.intel.com/content/www/us/en/ark/products/codename/74979/ice-lake.html) | Production | 3.0 | 1.3 | Yes
[Tiger Lake](https://ark.intel.com/content/www/us/en/ark/products/codename/88759/tiger-lake.html) | Production | 3.0 | 1.3 | Yes
[Rocket Lake](https://ark.intel.com/content/www/us/en/ark/products/codename/192985/rocket-lake.html) | Production | 3.0 | 1.3 | Yes
[Alder Lake](https://ark.intel.com/content/www/us/en/ark/products/codename/147470/products-formerly-alder-lake.html) | Production | 3.0 | 1.3 | Yes
[Meteor Lake](https://ark.intel.com/content/www/us/en/ark/products/codename/90353/products-formerly-meteor-lake.html) | Production | 3.0 | 1.3 | Yes
[Elkhart Lake](https://ark.intel.com/content/www/us/en/ark/products/codename/128825/elkhart-lake.html) | Production | 3.0 | -- | Yes
[Raptor Lake](https://ark.intel.com/content/www/us/en/ark/products/codename/215599/products-formerly-raptor-lake.html) | Production | 3.0 | 1.3 | Yes
[Broadwell](https://ark.intel.com/content/www/us/en/ark/products/codename/38530/broadwell.html) | Maintenance| 3.0 | -- | --
[Apollo Lake](https://ark.intel.com/content/www/us/en/ark/products/codename/80644/apollo-lake.html) | Maintenance| 3.0 | -- | --
[Gemini Lake](https://ark.intel.com/content/www/us/en/ark/products/codename/83915/gemini-lake.html) | Maintenance| 3.0 | -- | --

All platforms were validated on Ubuntu 22.04 LTS with stock kernel, unless noted otherwise.
- DG1 were tested with kernel 6.3.1-060301-generic and DG2 with 6.7.5-060705-generic
- Meteor Lake was tested with [kernel v6.6.7](https://kernel.ubuntu.com/mainline/v6.6.7/)
 
WSL support was tested with Windows host driver [101.5333](https://www.intel.com/content/www/us/en/download/785597/816901/intel-arc-iris-xe-graphics-windows.html), or [101.2114](https://www.intel.com/content/www/us/en/download/19344/intel-graphics-windows-dch-drivers.html) (ICL, EHL, CFL)
 
## Quality levels

- Experimental - no quality expectations
- Early support - platform may not be available on the market yet
- Pre-Release - suitable for integration and testing, gathering customer feedback
- Beta - suitable for integration and broad testing
- Production - Beta + meets API-specific conformance requirements; suitable for production
- Maintenance - Reduced test frequency compared to Production, no longer recommended for new deployments. Reactive support for customers.

## Additional information

- packages were built with custom flags NEO_ENABLE_i915_PRELIM_DETECTION=1

## Important changes
#710 was resolved by 420e1391b228586efa8546db343e8e6eb50e398b

## Changelog

https://github.com/intel/compute-runtime/compare/24.09.28717.12...24.13.29138.7

