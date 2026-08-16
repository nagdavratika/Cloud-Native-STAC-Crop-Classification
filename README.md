# Cloud-Native STAC Satellite Pipeline for Field Parcel Crop Classification

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Domain: Geoinformatics & Agriculture](https://img.shields.io/badge/Domain-Geoinformatics%20%7C%20Agriculture-green.svg)](#)
[![Stack: GeoPandas | STAC | XGBoost](https://img.shields.io/badge/Stack-GeoPandas%20%7C%20STAC%20%7C%20XGBoost-orange.svg)](#)
[![Data: Microsoft Planetary Computer STAC](https://img.shields.io/badge/Data-Copernicus%20Sentinel--2%20L2A-blueviolet.svg)](#)

A scalable, cloud-native geospatial machine learning pipeline designed to classify agricultural crop types across vector parcel boundaries. The platform queries SpatioTemporal Asset Catalog (STAC) endpoints on Microsoft Planetary Computer, computes multi-spectral biophysical indices ($NDVI$, $NDRE$, $NDWI$, $EVI$), extracts cadastral zonal signatures via `GeoPandas` and `Shapely`, and classifies crop types using an optimized `XGBoost` gradient-boosted engine.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Problem Statement & Background](#problem-statement--background)
- [System Architecture](#system-architecture)
- [Mathematical Methodology](#mathematical-methodology)
- [Spectral Indices Formulation](#spectral-indices-formulation)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Execution & Usage](#execution--usage)
- [Benchmark Results & Performance](#benchmark-results--performance)
- [License](#license)

---

## Project Overview

Traditional remote sensing workflows require downloading hundreds of gigabytes of satellite scene tiles before performing raster cropping and pixel classification. This approach is computationally inefficient and impractical for regional or national agricultural inventories.

This project delivers a **Cloud-Native Geospatial Architecture**:
1. **Serverless STAC Integration:** Interfaces with SpatioTemporal Asset Catalogs (`pystac-client`, `odc-stac`) to access Sentinel-2 Level-2A Bottom-Of-Atmosphere (BOA) reflectance bands on-demand without downloading full scene granules.
2. **Object-Based Parcel Analysis:** Replaces noisy pixel-based classifications with object-based zonal aggregation across cadastral farm parcel boundaries using `GeoPandas` and `Shapely`.
3. **Multi-Spectral Biophysical ML:** Ingests red-edge, near-infrared, and short-wave infrared dynamics into an `XGBoost` multiclass classifier, achieving high discriminatory power between spectrally similar crop varieties.

---

## Key Features

- **Cloud-Native STAC Protocol:** Queries cloud-optimized GeoTIFFs (COGs) with dynamic spatio-temporal and cloud-cover filtering ($< 10\%$).
- **Multi-Band Biophysical Index Suite:** Automatically derives Normalized Difference Vegetation Index ($NDVI$), Red-Edge Vegetation Index ($NDRE$), Normalized Difference Water Index ($NDWI$), and Enhanced Vegetation Index ($EVI$).
- **Vectorized Zonal Extraction:** Uses `GeoPandas` and `Shapely` for spatial overlay and zonal statistics aggregation over vector farm geometries.
- **Multiclass Classification Engine:** Optimized `XGBoost` classifier capable of distinguishing Corn, Soybeans, Alfalfa, and Fallow fields.
- **Spatial Validation:** Implements stratified parcel partitioning to evaluate model generalization without spatial leakage.

---

## Problem Statement & Background

Agricultural parcel classification faces key technical hurdles:

1. **Spectral Confusion Among Crop Phenologies:** Crops like corn and soybeans share similar optical reflectance during peak vegetative growth. Incorporating red-edge (Band 8A) and short-wave infrared (Band 11) indices is essential to separate chlorophyll saturation from canopy water content.
2. **Data Inefficiency in Legacy GIS:** Processing raster tiles locally creates large storage bottlenecks. Utilizing cloud-native STAC queries and object-based vector aggregation enables direct zonal feature modeling without redundant I/O operations.

---

## System Architecture

```text
  ┌─────────────────────────────────┐        ┌──────────────────────────────────┐
  │   Cadastral Vector Farm Parcels │        │  Microsoft Planetary Computer    │
  │     (GeoPandas GeoDataFrame)    │        │    (Sentinel-2 L2A STAC API)     │
  └────────────────┬────────────────┘        └─────────────────┬────────────────┘
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │    STAC Spatio-Temporal Query Engine      │
                   │      (B2, B4, B8, B8A, B11 COG Assets)    │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │    Biophysical Indices Computation        │
                   │        [NDVI | NDRE | NDWI | EVI]         │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │     Zonal Vector Feature Aggregation      │
                   │    (Parcel-Level Spectral Signatures)     │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │   Stratified Parcel Train-Test Split      │
                   │            (75% Train / 25% Test)         │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │       XGBoost Multiclass Classifier       │
                   │     [Corn | Soybeans | Alfalfa | Fallow]  │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │   Crop Classification Map & Evaluation    │
                   └───────────────────────────────────────────┘
