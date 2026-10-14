# Dataset report — Explainable AI for District-Level Prediction of Medium-Range Weather Forecast Busts over India

Sanket · Smart India Hackathon 2026 · Problem Statement SIH26079 · Ministry of Earth Sciences

D N Harith (1DS24AI012) · Dhanush Thota (1DS24AI057) · Guided by Dr. Kusumika Krori Dutta

*Generated from `build_report.py`; the PDF is the deliverable and this markdown is the same
content in a reviewable form.*

---


## Title of the Paper to Be Published

**Working title:** Explainable AI for District-Level Prediction of Medium-Range Weather Forecast Busts over India

## Problem Statement

An operational centre already issues a medium-range forecast for every district, and statistical post-processing already removes the errors a model makes consistently. What survives correction is the day-to-day failure: the forecast that is badly wrong because of the particular weather pattern of that day. Following Rodwell et al. [2], this project calls such a forecast a **bust** — an error that lands in the tail of that variable’s own historical error distribution, above the 90th percentile computed on training data only. The literature diagnoses continental upper-air busts after the event [2], [3], verifies Indian rainfall and circulation at regional scale [6]–[9], or explains the weather quantity a model predicts rather than the model’s own error [16]–[18]. The missing integration is an operationally usable, calibrated and locally explained probability that a specific medium-range forecast will fail, for several surface variables at once, at district scale over India.
The problem addressed in this assignment is the **data part** of that study. It is not a new weather forecast and does not claim to be one. The dataset must make a forecast archive and a verifying truth directly comparable at district scale, span enough years that rare failures are learnable, carry the predictors that hold the predictability signal (ensemble spread, run-to-run jumps, tendencies, season, geography and the Madden–Julian Oscillation), and be clean enough that no evaluation performed on it can be accused of leakage. The dataset described below is the result: NOAA GEFSv12 reforecast for the forecast side, ERA5 reanalysis for the truth, GADM 4.1 districts as the common spatial unit, and the static and dynamical covariates that surround them.

## Data Life Cycle

The full life cycle that the project follows is shown below. Stages 1 to 5 are complete for the archived dataset and are the subject of this document; stage 6 is implemented for the model now served and continues with every retrain; stages 7 to 9 follow the dataset submission.

**Table 1. Data life cycle**


| # | Stage | What is done | Status |
|---|---|---|---|
| 1 | Problem definition | Research gap and objective stated (above) | Done |
| 2 | Data acquisition | 12 GEFSv12 variables and 5 members per 00 UTC cycle from the AWS reforecast archive; ERA5 from the Copernicus CDS; 666 GADM 4.1 districts and their weight table; elevation, land–sea masks, MJO index | Done |
| 3 | Data description and exploration | Types, schema, counts, missingness, error distributions, coverage by lead day and district | Done |
| 4 | Validation and cleaning | Completeness, duplicate, plausibility-range, alignment, leakage and consistency checks (Section 2) | Done |
| 5 | Preprocessing, splitting and storage | District area-weighting through one weight table; daily aggregation to UTC days; chronological split with a Day-10 embargo; hive-partitioned Parquet plus SQLite lineage | Done |
| 6 | Model building | One XGBoost error regressor per variable; a bust classifier trained on their out-of-fold error evidence; TreeSHAP attributions; a CNN challenger over the gridded fields | Discharging; repeats on each retrain |
| 7 | Baselines and calibration | Climatology, lead-day, ensemble-spread, EMOS, IDR and analog baselines scored on the same held-out rows; Platt calibration; promotion gate | In progress |
| 8 | Evaluation | Brier skill, ROC-AUC, SEDI, CORP reliability decomposition, relative economic value, conformal coverage, SHAP stability | In progress |
| 9 | Reporting and publication | Results, comparison with the baseline ladder, paper | To follow |

## 1. DATA DESCRIPTION


### 1.1 Type of data

The dataset has four linked parts, and all four are numeric. There is a **forecast archive** (NOAA GEFSv12 reforecast: one 00 UTC cycle a day, 5 ensemble members, 0.25° global grid, 3-hourly steps reduced to daily fields, lead days 1 to 10); a **verifying observation archive** (ERA5 reanalysis, on the same 0.25° grid and the same UTC day convention); a **geography layer** (666 GADM 4.1 districts, the grid-cell weight table that connects districts to cells, and per-district descriptors derived from the geometry); and **covariate files** (a daily Madden–Julian Oscillation index, an elevation grid for every cell the weight table uses, and ERA5 and GEFS land–sea masks). Forecasts and observations are not kept as two differently shaped products: both are reduced to the same canonical row — one district, one variable, one valid date, one value type, one ensemble member and lead day, one value — and it is the paired canonical row that every model reads. There is no text, image, or free-form data anywhere in the training path.
What is predicted is not a weather value but the **error between the two sides**. Each forecast row is paired with the observation for the same district, variable and UTC day, and the absolute error of the pair is carried with it. The label is derived from those errors: a bust is an error above the 90th percentile of that variable’s own error distribution, and the percentile is computed on training cycles only, never on the rows it will be scored against. Eight surface variables are carried (Table 4). Wind direction is an input but deliberately not a label variable: a daily resultant direction is mostly noise when the wind is light, and 86.5% of its apparent busts under plain angle arithmetic disappeared once the error was measured the short way round the circle.
The data is grouped and correlated in space and in time rather than being a set of independent samples. Consecutive days at one district are strongly autocorrelated, neighbouring districts share 0.25° cells through the weight table, and the five members of a cycle are perturbations of one analysis rather than five observations of the atmosphere. The chronological split, the 30-day out-of-fold blocks and the Day-10 embargo described in Section 1.4 exist because of that structure, not in spite of it.

**Table 2. Attribute types in the canonical store (one row = one district, variable, valid date, value type, member and lead)**


| Attribute type | Attributes | Count |
|---|---|---|
| Identifier (text) | record_id, upload_batch_id, source_file, source_column | 4 |
| Categorical (nominal) | variable (8 levels), value_type (forecast, observed), grain, verification_status (final, provisional) | 4 |
| Numeric, continuous | value, lat, lon, mapping_confidence | 4 |
| Timestamp | init_date, valid_date, ingested_at | 3 |
| Categorical (region) | region_id, region_name (666 levels), region_resolution_method | 3 |
| Numeric, discrete (integer) | lead_time_days (1 to 10) | 1 |
| Categorical (member) | ensemble_member_id (c00, p01–p04 in the archive; up to 31 in the operational feed) | 1 |
| **Total** |  | **20** |

### 1.2 Number of datasets and their sources

There are four source families, fetched by their own scripts under `backend/scripts/`, and ten products tracked in the repository. The bulk of the archive — the twenty-year reforecast — is fetched by CI and stored as a release asset rather than committed: at 0.81 TB for a full year at daily density, it does not belong in git. Every row records where it came from, so any value can be traced back to the file and the message that produced it: the reforecast carries the GRIB2 object path and message number for each of its twelve variables, and the observation side carries the product and API that served it. **No value in this dataset is synthetic, simulated, interpolated or placeholder.** Where a number could not be fetched, there is no row.

**Table 3. Datasets, their sources and their terms**


| Dataset | Source | Used for | Term |
|---|---|---|---|
| GEFSv12 reforecast (2000–2019) | NOAA GEFSv12 reforecast archive, AWS Open Data bucket `noaa-gefs-retrospective`; one 00 UTC cycle a day, 5 members (c00, p01–p04), 0.25°, 3-hourly to 240 h [4], [26] | Training and calibration forecasts | Public domain (NOAA Open Data) |
| GEFS operational feed | NOAA NOMADS and its AWS mirror, pulled every six hours; up to 31 members | Live scoring, Replay and the alerts feed | Public domain |
| ERA5 reanalysis | Copernicus Climate Change Service, Climate Data Store, dataset `reanalysis-era5-single-levels` [20], [25] | Observations for training, verification and the bust label | CC-BY 4.0 |
| ERA5 recent days | Open-Meteo Historical Weather API with `models=era5`, answered from ERA5’s own 0.25° cell | Observations for days the CDS archive lags | CC-BY 4.0 |
| IMD gridded rainfall | India Meteorological Department, Pune, 0.25° gauge-based daily rainfall | Planned rainfall truth (the served run’s rainfall truth is ERA5) | IMD terms, credit required |
| District boundaries | GADM 4.1 admin-2, 666 districts, with three deliberate corrections (Ladakh reassigned out of Jammu & Kashmir, disputed-territory features dissolved into their parent districts, Delhi relabelled) [24] | Map geometry, the weight table, and every district descriptor | Free for academic and non-commercial use |
| Cell elevation | Open-Elevation public API, serving CGIAR-CSI SRTM 250 m, for the 4,902 cells the weight table uses [22] | Elevation descriptor and the CNN’s terrain channel | Acknowledge CIAT |
| Land–sea masks | ERA5’s own land–sea mask, and GEFS sea cells identified from the archived gridded soil-moisture fields | Refusing to compare soil moisture over the sea | CC-BY 4.0 / public domain |
| MJO index | NOAA PSL OLR-based MJO index (OMI), 1991 to present, transformed to the RMM1 / RMM2 convention by PSL’s own documented mapping [21] | MJO features (dropped from new runs while the file’s tail is stale) | NOAA open data |
| Verification-product comparison | MERRA-2 through the NASA POWER daily API, paired against ERA5 [23] | Measuring the ground truth’s own uncertainty | NASA open data |
Two details of the acquisition matter for reading the rest of this document. First, only the GRIB2 messages that are needed are downloaded, by HTTP range requests keyed off each file’s `.idx` sidecar: 1.79 GB per initialisation for 9 variables and 2.2 GB for 12, where the file holding just one of those fields is 296 MB on its own, of which 19 MB is read. Second, the twelve variables are not all of the same kind: nine have an observed counterpart and are paired; three — 500 hPa geopotential height, surface CAPE and surface CIN — have no observation to verify against, never enter the paired store, and reach only the convolutional challenger as grid channels.

**Table 4. The eight predictand variables and the three grid-only predictors**


| Variable (canonical name) | Unit | GEFSv12 field | Leads held | Bust label |
|---|---|---|---|---|
| temperature_c | °C | tmp_2m (t2m) | Day 1–10 | Yes |
| humidity_pct | % RH | spfh_2m (sh2), converted to RH after the district mean on both sides | Day 1–10 | Yes |
| rainfall_mm | mm/day | apcp_sfc (tp), accumulated over the model day | Day 1–10 | Yes |
| pressure_hpa | hPa | pres_msl (msl) | Day 1–10 | Yes |
| atmospheric_ moisture_kgm2 | kg/m² (TCWV) | pwat_eatm (pwat) | Day 1–10 | Yes |
| wind_speed_ms | m/s | ugrd_hgt / vgrd_hgt (u10, v10) | Day 1–5 only | Yes |
| wind_direction_deg | degrees (from) | ugrd_hgt / vgrd_hgt (u10, v10) | Day 1–5 only | No — input only |
| soil_moisture_pct | % volumetric | soilw_bgrnd (soilw), top layer | Day 1–3 only | Yes |
| Z500 (hgt_pres_abv700mb) | m | gh at 500 mb | Day 1–10 | No — CNN channel |
| CAPE (cape_sfc) | J/kg | cape | Day 1–10 | No — CNN channel |
| CIN (cin_sfc) | J/kg | cin | Day 1–10 | No — CNN channel |
The lead-day limits in that table are the archive’s, not the project’s: the GEFSv12 reforecast carries 10 m wind only to 120 h and soil moisture only to 72 h, so no model can have seen those variables beyond those leads. The serving layer enforces the same limits rather than scoring a variable at a lead its model never trained on.

### 1.3 Missing data

Missing data takes two forms here, and they are treated differently. In the canonical store a value that does not exist is written as an **absent row**, not as a null cell: the store is normalised on (district, variable, valid date, value type, member, lead), so an incomplete source cannot leave a half-filled record behind. The ingest refuses a cycle it cannot complete rather than storing part of one, because a short rainfall *sum* is roughly half the real accumulation and rainfall drives most busts. Where nulls do appear, they are structural, and the contract that every frame is checked against names each one and why it is allowed.
Measured on the two archived 2019 sample files that ship with the repository, whose numbers are reproducible from the files themselves: the reforecast file holds 30,600 rows × 36 columns = 1,101,600 cells, of which 156,060 (14.2%) are null. The nulls are exactly two variables — 10 m wind speed and direction, held only to Day 5 (15,300 of 30,600 rows each, 50.0%), and soil moisture, held only to Day 3 (21,420 of 30,600, 70.0%). The equality is exact in both cases: the gap is a property of the archive’s lead coverage, not of any particular cycle, and the coverage is identical for all 17 sampled cycles and all five members (Figure 2a). The ERA5 observation sample has 13,680 rows × 16 columns = 218,880 cells with **0 missing**: every one of its 36 stations reports on all 380 days (1 January 2019 to 15 January 2020). The MJO index has 12,959 daily rows from 1 January 1991 with 0 missing and no date gaps, and its earliest date is the only boundary: it is joined to a forecast cycle as of the issue date with a five-day tolerance, so cycles before 1991 or after a stale tail simply carry no MJO value. Every one of the 666 district rows has a complete descriptor row, and every one of the 666 weight vectors sums to 1.000000.
The second form of missingness is at the level of the archive rather than the cell: 17 of the 7,305 daily reforecast cycles of 2000–2019 (0.23%) are absent from the store — ten days of 2004, five of 2006 and two of 2008 — and training and scoring simply have no events on those dates. Nothing is imputed to cover them, and re-fetching them is an open fetch-once question rather than a quiet fix. Two further member-days (p03 of 2004-03-05 and p01 of 2019-02-09) carried values that cannot be weather rather than missing ones; those were dropped by the plausibility check described in Section 2.

**Table 5. Attributes by level of missing data (measured on the files in `backend/data/`)**


| Missing-data level | Attributes | Records affected |
|---|---|---|
| 0% (complete) | All 20 canonical-store columns; all 16 observation columns; 26 of the 36 reforecast columns; the MJO index, the weight table, the descriptors and both land–sea masks | 0 nulls in the canonical store; 0 of 218,880 observation cells; 945,540 of 1,101,600 reforecast cells (85.8%) |
| up to 50% | 10 m wind speed and 10 m wind direction (present on Day 1–5 only) | 15,300 of 30,600 rows each (50.0%), identical for every cycle and member |
| 70% | Soil moisture (present on Day 1–3 only) | 21,420 of 30,600 rows (70.0%) |
| more than 70% | None | 0 |
| **Total** | **36 reforecast columns, 16 observation columns** | **156,060 null cells, all of them structural** |

### 1.4 Common attributes

All 20 canonical columns (Table 2) are common to every source, whatever format it arrived in: GRIB2 and NetCDF from the archives, CSV and XLSX from an upload, JSON for elevation, and zipped NetCDF pairs from the CDS. The parsers are format-agnostic and the schema mapper matches incoming columns to the canonical schema with a confidence score, but the schema itself is frozen: the Arrow schema is asserted equal to the field list of the canonical row model at import time, so a drift between the two fails loudly at once instead of quietly downstream.
Units are canonical and identical on both sides of every label: mm/day for rainfall over the model day, °C for temperature, %RH for humidity, hPa for mean-sea-level pressure, kg/m² for column water vapour, % volumetric for soil moisture, and m/s and degrees (meteorological, from-direction) for wind. Derived quantities are derived the same way on both sides — relative humidity from specific humidity, temperature and pressure *after* the district mean; wind speed and direction from the two components — so a forecast row and a truth row are never two different quantities under one name. An accumulation is stamped by the end of the hour it covers, so every variable is grouped by `valid_time − 1 h`; that makes an observation day the half-open window (t−24 h, t], which is deliberately the same window the forecast side uses for day k, ((k−1)×24, k×24]. Both sides of a bust label therefore describe the identical interval.
The spatial convention is common too, and it is the one design decision the whole dataset rests on: **there is exactly one weight table**. Each district’s value is the area-weighted mean of every 0.25° cell its polygon overlaps — 10,373 district–cell pairs over 4,902 distinct cells, a median of 14 cells per district (Figure 3b) — rather than the value of the nearest grid point. That is not a refinement: ten districts, among them Kolkata, Hyderabad, the Puducherry enclaves and Lakshadweep, contain no grid centre at all, and for them a nearest-point sample would mean reporting a cell that lies outside the district it is named after. Forecasts and observations go through the same table, so both sides of the label are area means over the same polygon. The same attribute structure is also common across the splits (Table 6) and across the eight variables: the frames differ in which cycles they hold, never in their columns.

**Table 6. Split composition of the training data**


| Split | Cycles | Years | What it is used for |
|---|---|---|---|
| train | 6,193 daily cycles | 2000–2016 | Bust thresholds; the per-district, per-variable, per-lead, per-month bias table; historical bust frequency; the regressors and the classifier |
| validation | Held out of training | within 2000–2016 | Early stopping, Platt calibration and tuning (so validation scores are optimistic, and are never quoted as test results) |
| test (held out) | 365 daily cycles | 2017 | Every metric the site publishes; the promotion gate; the baseline ladder. 2018 and 2019 sit in the archive unused, and are the next evaluation |
| embargo | 10 days either side of every boundary | — | No observed day appears on both sides of a split or of an out-of-fold block; out-of-fold blocks are 30-day spans of forecast cycles |

## 2. DATA CLEANING METHOD SELECTION AND JUSTIFICATION

This dataset is fetched from other people’s archives rather than authored by the team, so its cleaning is not about repairing typing errors. It is about the places where two individually good products will still disagree or mislead: a forecast field that has no observed counterpart, two models that fill the sea oppositely, a truth product that arrives one day late, an input that quietly uses the answer. Each candidate problem below was checked against the data, and a treatment was chosen only where a check found something. The treatments were chosen because the obvious alternatives would damage the dataset: imputing a missing forecast step fabricates the accumulation that drives most busts, deleting extreme errors removes exactly the events the label is built on, and subtracting a per-district offset without recording it hides the difference between a biased district and a failing forecast.

**Table 7. Cleaning and validation methods selected, and why**


| Issue | Finding | Method selected | Justification |
|---|---|---|---|
| Missing values | No null cells in the canonical store or in the observation sample; 14.2% of the reforecast sample’s cells are the documented structural gaps (wind beyond Day 5, soil moisture beyond Day 3) | Absence of a row rather than a null cell; a completeness guard refuses a cycle that is short of its 33,300 rows; no imputation anywhere in the pipeline | A short rainfall sum is about half the real accumulation, and a fabricated member would enter the ensemble spread as evidence |
| Duplicates | Re-ingesting a source leaves two rows for one key. Found on real data: IMD rainfall batches ingested after the ERA5 batches won the dedupe and replaced 2016–2019 rainfall truth in the training store | Dedupe on (region, valid_date, variable, value_type, init_date, lead, member, source_column) with the newest non-provisional row winning; quarantined and retired batches are refused before any frame is built, and a cached year is rebuilt when its batch signature changes | A duplicate is not a repeated row here, it is a different truth product: it changes the label silently and inflates any score computed on it |
| Implausible values (outliers) | Two corrupt member-days in twenty years: p03 of 2004-03-05 with Day-2 2 m temperature from −23,380,072 to +21,808,336 °C in all 666 districts (other members −18.5 to 32.4); p01 of 2019-02-09 with Day-1 column water vapour of 12,500–12,546 kg/m² (others 0.7–55.3) | Range check against a per-variable plausible range, applied at the head of the one frame-building path that training, scoring, live and Replay all share; a corrupt member is dropped and the ensemble is the members left (666 rows on each affected cycle, none on a clean one) | One such member put that day’s temperature RMSE at 34,612 °C. A plausible range is not a smoothing filter: real extremes stay, because the label lives in the tail |
| Data leakage | A time-lagged error input existed that no live forecast can have (it reads the previous lead’s realised error of the same forecast), and 2016–2019 rainfall truth came from a one-day-late IMD file that had won the dedupe | Observation-derived columns are refused as model inputs (a test enforces it); the four IMD batches were quarantined and the store restored to ERA5; thresholds, bias tables and calibrators are fitted on training cycles only; the split is chronological with a Day-10 embargo | A held-out score must be the score live scoring would produce. The leak was found by an audit after the model was served, and the model it affected is replaced by the next retrain rather than defended |
| Skewed labels (rainfall) | Rainfall is zero on 44.8% of observation days and its error is heavy tailed: median 1.53 mm/day, 90th percentile 14.38, 99th 48.14, maximum 170.8 mm/day in the archived sample | A percentile definition of a bust — the variable’s own training-set 90th percentile — instead of a fixed threshold; zero-inflation reported with the variable rather than corrected by resampling | A fixed millimetre threshold would label almost nothing in a dry district and almost everything in the monsoon; resampling dry days would repeat near-identical rows instead of adding information |
| Skewed labels (steady bias) | 64% of squared error for temperature and humidity, and 90% for soil moisture, is a fixed per-district offset that ordinary bias correction removes; rainfall is 6% and its error grows with lead as forecast failure should | Busts are defined on bias-corrected error: the training-period mean error per (district, variable, lead, month) is removed first, with backoff to the district’s month across leads and then to its year when a cell holds fewer than 30 events; a key that is thin at every level is not compared at all | Otherwise a “bust” means “this district is always off”, which NCMRWF’s own post-processing already removes — the failure that remains is the one worth predicting |
| Inconsistent format or units | Inputs arrive as GRIB2, NetCDF, CSV, XLSX and JSON; ERA5 stamps an accumulation with the end of its hour; humidity is specific humidity on the forecast side and relative humidity on the observation side; wind arrives as two components; the CDS returns a zip of two NetCDF files split by step type | Format-agnostic parsers into one canonical schema with confidence-scored column matching; accumulation grouped by valid_time − 1 h; relative humidity derived from specific humidity, temperature and pressure after the district mean on both sides; speed and direction derived from u and v; the CDS zip is unzipped and the two files read separately | One schema and one set of units is what makes a forecast row and a truth row comparable at all; deriving RH after the mean on both sides keeps a bounded quantity from being averaged as a ratio |
| Misplaced or lost geography | For eight 2017 districts the schema mapper did not recognise the region column and placed rows by coordinate; five of them vanished and three held a neighbour’s values. Separately, ten districts contain no 0.25° grid centre at all | One area-overlap weight table used by both sides; rows are keyed by district identifier, not by position; the mapper is fixed and the affected 2017 districts are re-ingested | A nearest-point sample cannot represent a district that contains no grid centre, and a mislabelled district corrupts both its own rows and its neighbour’s |
| Sea read as land (soil moisture) | GEFS writes the sea as saturated soil (~98% volumetric, median 99.78) while ERA5 writes it as 0. Paired through the weight table this labelled four districts (Nicobar Islands, Lakshadweep, Diu, Mumbai City) a soil-moisture bust on 100% of their rows against 0.9% elsewhere, and the effect is a gradient in how much sea each coastal district contains | A per-district land–sea mask built from ERA5’s own mask and from GEFS sea cells identified in the archived gridded fields; soil moisture is dropped on both sides for any district more than 5% sea in either product, and the exclusion list is required to exist rather than defaulted | Masking is honest and imputation is not: a district that is mostly sea has no soil-moisture value to compare, and leaving the pair in teaches a model a district-identity shortcut |
| Incomplete cycles | NOAA drops forecast steps mid-pull; a short rainfall sum silently halves the quantity that drives most busts, and a partial pull can leave an observation fetch half-done | A completeness check per cycle refuses an incomplete cycle rather than publishing part of one, with an S3 byte-range fallback that refills missing messages before the daily reduction; observation pulls report completeness rather than assuming it | Publishing nothing for a day is recoverable; publishing a thin cycle teaches the model that a real rain day was dry |
| Scaling | Not applied to the stored data, and not applied to the tree models | None for the stored dataset or for XGBoost, which is invariant to a monotone rescaling of a feature; the convolutional challenger normalises its channels with training statistics; the classifier’s probabilities are mapped to observed frequencies by a Platt fit on validation, with a non-positive slope refused | Rescaling the stored values would add a version to every file and change nothing the served model reads; calibrating the probability is needed, because an uncalibrated classifier’s lowest reliability bin predicted 0.158 against ~0.07 observed |

## 3. RESULTS


### 3.1 Verification results

Every cycle is verified before it is stored, and the check is structural rather than statistical. At district scale one cycle is 666 districts × 5 members × 10 lead days = 33,300 wide rows, and a cycle that does not match that shape is refused rather than written short. Measured end to end on 2017: **365 cycles, 12,154,500 rows, zero refusals, every cycle exactly 33,300 rows**. Across 2000–2019 the store holds 7,288 of the 7,305 daily cycles (99.77%), the gaps being the 17 dates listed in Section 1.3.
The two files reproduced for this report were verified the same way and pass the same checks. The reforecast sample has 30,600 rows = 17 cycles × 36 stations × 5 members × 10 lead days, with 1,800 rows on every cycle and no cycle short. The observation file has 13,680 rows = 36 stations × 380 days with no null in any of its 16 columns. Most important for pairing, the two files name the same 36 stations, so all 6,120 archived forecast rows have a truth for the same district and UTC day — the join that a bust label is built on is lossless here, as it is at district scale, where the forecast and the truth are reduced through the same weight table.
Two alignment results deserve to be stated because the dataset is unusable if they fail. First, dates: IMD’s gauge day runs 0830–0830 IST, so IMD’s date D is model day D−1. A join that treated them as the same day put every IMD value one model day late. Settled from data rather than from either documentation: the Spearman correlation of IMD district rainfall on IMD date D against independent reanalysis rainfall summed to UTC day D+k peaks at k = −1 for 8 of 8 districts against ERA5 and for the same 8 of 8 against MERRA-2 (means 0.698 and 0.810, against 0.556 and 0.621 at k = 0), and the merge now moves every IMD date back one day and refuses to write a merge whose rainfall does not correlate best with ERA5 at lag 0. Second, product agreement: ERA5 and MERRA-2 are close on average, and a comparison of 21,492 paired city-days measured bias to within 0.06 °C, 2.4 %RH, 0.1 mm/day and 0.75 m/s. The same comparison is what measures the truth’s own uncertainty, below.

**Table 8. Forecast–observation error statistics of the archived 2019 paired sample (6,120 district-day pairs; the 90th percentile is the bust threshold under the training-only rule)**


| Variable | Unit | Bias | MAE | Median | p90 (train.) | p99 | Maximum |
|---|---|---|---|---|---|---|---|
| temperature_c | °C | +0.64 | 1.81 | 1.25 | 4.16 | 8.54 | 17.34 |
| humidity_pct | %RH | −9.04 | 11.11 | 9.11 | 23.57 | 37.57 | 49.29 |
| rainfall_mm | mm/day | +0.25 | 5.30 | 1.53 | 14.38 | 48.14 | 170.81 |
| pressure_hpa | hPa | −0.62 | 1.31 | 0.88 | 2.98 | 6.37 | 19.69 |
| atmospheric_ moisture_kgm2 | kg/m² | −1.30 | 3.20 | 2.33 | 7.00 | 14.27 | 34.95 |
| wind_speed_ms | m/s | +0.71 | 0.97 | 0.66 | 2.19 | 4.77 | 10.37 |

Wind is paired on 3,060 of the 6,120 pairs: the archive holds it only to Day 5. The humidity bias (−9.0 %RH, the forecast drier than ERA5) is the bias that motivated per-district, per-lead bias correction before the bust label is defined.
Table 8 shows the shape of each variable’s error, and two things stand out. Rainfall is the hardest and the most skewed: nearly half of its observation days are dry, its median error is 1.5 mm/day, and its 99th percentile is 32 times that, with 16.6% of events above the threshold — the tail that matters is long and it is where the label lives. Humidity carries a systematic dry bias of about 9 %RH, so more of its ‘large errors’ are a standing offset than a forecast failure. That distinction is measurable, and it is what the bias-corrected label version exists to separate.

**Table 9. Absolute error by lead day (MAE of the ensemble mean against ERA5, archived 2019 sample)**


| Variable | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|---|---|---|
| temperature_c (°C) | 1.70 | 1.77 | 1.74 | 1.83 | 1.80 | 1.84 | 1.91 | 1.83 | 1.87 | 1.86 |
| rainfall_mm (mm/day) | 4.59 | 4.60 | 5.11 | 5.11 | 4.82 | 5.31 | 5.71 | 6.59 | 5.53 | 5.64 |
| humidity_pct (%RH) | 10.8 | 10.8 | 10.6 | 10.7 | 10.6 | 10.9 | 11.6 | 11.7 | 11.8 | 11.4 |
| wind_speed_ms (m/s) | 0.82 | 0.91 | 1.02 | 1.03 | 1.05 | — | — | — | — | — |

Rainfall error grows with lead, as a forecast failure should, and roughly doubles from Day 1 to Day 8. Wind stops at Day 5 because the archive does. Temperature and humidity barely grow at all, which is the signature of steady per-district bias rather than of a forecast coming apart.

### 3.2 Label statistics

The stored quantity is a continuous error and the label is its binary tail: a bust is an error above the training-set 90th percentile for that variable. So the distribution in Figure 1 *is* the label’s support, and the dashed line in each panel is the threshold it will be cut at. The distributions are all right-skewed, which is what makes a percentile definition necessary rather than convenient: for temperature the threshold is 4.16 °C against a median error of 1.25, while for rainfall the threshold is 14.4 mm/day against a median of 1.5, and the same millimetre figure would mean something completely different in a desert and in Kerala.
Two rates are worth stating, and they are different numbers for a reason. In this in-sample illustration the thresholds are quantiles of the same 6,120 rows they label, so each variable busts on exactly 10% of them by construction; an event (any variable) busts on 38.6% of the 6,120 paired rows. In production the thresholds and the bias table are fitted on training cycles only and the label is computed on bias-corrected error, and the published event base rate on held-out data is **43–49%** depending on the run and on how many variables exist at that lead. The reason is structural: an event busts if *any* of about eight variables is over its own threshold, so 1 − 0.9<sup>8</sup> ≈ 57% before dependence between the variables, and fewer than eight variables are available from Day 4 and Day 6. Any analysis that assumed a 10% base rate would mis-weight the problem; that is precisely why the served classifier is not re-weighted for rarity.

![Figure 1. Absolute error of the GEFSv12 ensemble mean against ERA5 for the six continuous pairings of the archived 2019 sample (6,120 pairs, wind 3,060). The dashed line is the variable’s own 90th percentile, which is the bust threshold; the clipped right-hand bar in each panel collects the tail beyond the 99.5th percentile.](assets/fig1_error_distributions.png)

*Figure 1. Absolute error of the GEFSv12 ensemble mean against ERA5 for the six continuous pairings of the archived 2019 sample (6,120 pairs, wind 3,060). The dashed line is the variable’s own 90th percentile, which is the bust threshold; the clipped right-hand bar in each panel collects the tail beyond the 99.5th percentile.*

![Figure 2. Source coverage of the archived 2019 sample. (a) The non-null share of each variable by lead day: the two white blocks are the archive’s lead limits (10 m wind beyond Day 5, soil moisture beyond Day 3), identical for every cycle and member. (b) The 17 sampled initialisation dates of 2019: fortnightly through the monsoon, monthly otherwise, so each year samples the full seasonal cycle without pretending that neighbouring cycles are independent.](assets/fig2_coverage.png)

*Figure 2. Source coverage of the archived 2019 sample. (a) The non-null share of each variable by lead day: the two white blocks are the archive’s lead limits (10 m wind beyond Day 5, soil moisture beyond Day 3), identical for every cycle and member. (b) The 17 sampled initialisation dates of 2019: fortnightly through the monsoon, monthly otherwise, so each year samples the full seasonal cycle without pretending that neighbouring cycles are independent.*

![Figure 3. The static geography layer. (a) All 666 district centroids, coloured by mean elevation and sized by area; the Himalayan districts are both the largest and the highest, and elevation is a real driver of pressure error (Leh runs about ten times the typical district). (b) How many 0.25° cells each district’s polygon overlaps — the weight table in one picture. (c) Distance from each district to the outer edge of the dissolved landmass (coastline and international borders together).](assets/fig3_districts.png)

*Figure 3. The static geography layer. (a) All 666 district centroids, coloured by mean elevation and sized by area; the Himalayan districts are both the largest and the highest, and elevation is a real driver of pressure error (Leh runs about ten times the typical district). (b) How many 0.25° cells each district’s polygon overlaps — the weight table in one picture. (c) Distance from each district to the outer edge of the dissolved landmass (coastline and international borders together).*

### 3.3 Effect of the cleaning steps on the label distribution

The table compares each quantity before and after the treatment that targeted it. Every pair of numbers was measured on real data, and each is the reason the treatment exists rather than an argument for it. The last three rows are the ones that changed the label most: they are the difference between a label that measures forecast failure and one that measures a district’s permanent offset or a mis-scaled angle.

**Table 10. Before and after the cleaning steps (measured on real data)**


| Quantity | Before the treatment | After the treatment |
|---|---|---|
| 2004-03-05 Day-2 temperature RMSE, with the corrupt p03 member in | 34,612 °C | Within the plausible range: exactly 666 rows dropped on each affected cycle and none on a clean one |
| Rainfall truth for 2016–2019 in the live run’s validation and test years | One-day-late IMD-merged rainfall, differing from ERA5 in 72.5% of 2018 rows and 79.8% of 2019 rows | ERA5 CDS batches; the IMD batches are quarantined and any frame build that finds them now stops instead of choosing |
| Humidity bust threshold, Nov 2017, 666 districts | 27.6 %RH on raw error (a dry-biased forecast made an ordinary day look like a failure) | 10.4 %RH after per-district, per-lead bias correction |
| Wind-direction error p90, real CI sample | 191.0° under plain angle arithmetic (a northerly is a bust by construction; 86.5% of direction busts were this artefact) | 115.1° with the error taken the short way round the circle; ensemble spread p90 falls from 118.0° to 55.0° |
| Event bust rate by lead, leaving out each district’s own bias | 0.31 at Day 1 rising to 0.50 at Day 10 only after the offset is removed; correlation of lead with bust ~0.02 before | 0.25 at Day 1 to 0.50 at Day 10 with the bias removed leave-one-cycle-out (no in-sample fitting), and rainfall growing with lead throughout |
| Soil moisture for the four sea-covered districts | Labelled a bust on 100% of their paired rows against 0.9% for the other 662 districts — a guaranteed fabricated label | Not compared at all: 18 districts exceed the 5% sea threshold in GEFS or ERA5 and are excluded on both sides |
| Member count used for spread | 5 members a day in the archive (11 on Wednesdays) against 31 in the operational feed | Stated as a limitation rather than corrected: spread features remain a noisy estimate, and a time-lagged pool across earlier cycles is added alongside them |

## 4. CONCLUSION

The dataset is four linked, all-numeric parts on a single spatial unit: the NOAA GEFSv12 reforecast (2000–2019, one 00 UTC cycle a day, five members, 0.25°, twelve variables, Day 1–10) as the forecast side; ERA5 reanalysis, reduced through the same weight table, as the truth; GADM 4.1 as 666 districts with one area-overlap weight table of 10,373 district–cell pairs over 4,902 cells; and the covariates that surround them (elevation, two land–sea masks, the MJO index). Both sides are joined into one canonical 20-column row, the error of each pair is carried with it, and the label — a bust above that variable’s own training-set 90th percentile — is derived from it. It is complete: no null cells in the store, no nulls anywhere in the observation sample, no duplicate keys after dedupe, all 666 districts with a complete weight vector summing to 1.0, and 7,288 of 7,305 archive cycles present. The only missingness is structural and is exactly the archive’s own lead limits: 50% of rows for 10 m wind (Day 1–5) and 70% for soil moisture (Day 1–3).
The cleaning was chiefly verification, because the usual treatments would damage products that are correct in themselves: the real problems found were not missing values but two products disagreeing — a sea convention read as saturated soil, an IMD date convention one day out, an angle difference that made a northerly a bust, a truth file that won a dedupe it should have lost, and one input that used an observation from after issue time. Each was measured, treated, and where a trained model still carries the old behaviour the model is to be replaced rather than defended. The effect on the label is visible: the humidity threshold falls from 27.6 to 10.4 %RH, wind-direction error p90 from 191° to 115°, and the four sea-covered districts stop being guaranteed busts.
The main limitations are stated with the numbers, because they bound what the dataset can support. The truth is a reanalysis, not a gauge network: two leading products disagree per day by 24–43% of the bust threshold depending on the variable (rainfall 23.5%, humidity 36.5%, wind 38.0%, temperature 43.5% of the threshold over 21,492 paired city-days), and that disagreement is a floor under every label. Only five of the thirty-one operational ensemble members exist in the archive, so spread is a noisy estimate. Two variables cannot be scored at long lead at all. One held-out year (2017) has been evaluated so far; 2018 and 2019 are in the archive and unused. And the bust is defined on surface-variable error rather than on the synoptic 500 hPa criterion of Rodwell et al. [2] — deliberately, because surface error is what reaches agriculture and disaster response. The next stages of the life cycle are to complete the baseline ladder and calibration on the bias-corrected label, to evaluate on the held-out multi-year window, and to publish the results with the explanation stability that the literature survey requires.

## 5. REFERENCES

The companion document for this assignment is the project’s literature survey, `docs/literature-survey-explainable-ai-weather-busts.pdf`, which reviews the twenty studies the numbered references above correspond to and positions this dataset within them. The report below cites those references by the same numbers.
**Sources of the datasets.** Every number in this report traces to one of the sources below. Each was fetched by its own script in `backend/scripts/`, which records the source in the data it writes, and each keeps its own licence and attribution — the project’s MIT licence covers the project’s own code and documents only. Boundary geometry for a political map of India raises questions of its own, reviewed separately in the repository.
Forecast archive: NOAA GEFSv12 reforecast (2000–2019), AWS Open Data bucket `noaa-gefs-retrospective` — https://registry.opendata.aws/noaa-gefs-reforecast/ [26] and https://noaa-gefs-retrospective.s3.amazonaws.com (public domain). Live feed: NOAA NOMADS and its AWS mirror (public domain).
Observations: ERA5 reanalysis, Copernicus Climate Change Service (C3S) Climate Data Store, dataset `reanalysis-era5-single-levels` — https://cds.climate.copernicus.eu/datasets (CC-BY 4.0; “Generated using Copernicus Climate Change Service information”). Recent days are served through the Open-Meteo Historical Weather API with `models=era5` — https://open-meteo.com/ (CC-BY 4.0).
Madden–Julian Oscillation: NOAA PSL OLR-based MJO Index (OMI), daily, 1991 → present — https://psl.noaa.gov/mjo/mjoindex/omi.1x.txt, transformed to the RMM1/RMM2 convention by PSL’s own documented mapping (NOAA open data; cite Kiladis et al. [21]).
Ground-truth uncertainty: MERRA-2 through the NASA POWER daily point API — https://power.larc.nasa.gov/api/temporal/daily/point (NASA open data; cite NASA POWER [23]).
District elevation: Open-Elevation public API, serving the CGIAR-CSI SRTM 250 m product — https://api.open-elevation.com/api/v1/lookup (acknowledge CIAT; cite Jarvis et al. [22]).
District boundaries: GADM 4.1 admin-2 for India, 666 districts — https://gadm.org/ [24] (free for academic and other non-commercial use; redistribution or commercial use requires GADM’s prior permission). Claimed-territory outlines on the map: Natural Earth — https://www.naturalearthdata.com/ (public domain).
Rainfall truth (planned, and the truth for the IMD-verified years): India Meteorological Department gridded daily rainfall, 0.25° — https://imdpune.gov.in/cmpg/Griddata/rainfall.php (IMD terms; credit required).
**Works cited.** The numbered entries are the references of the corresponding literature survey, kept in the same order so the two documents can be read together.
[1] Potvin, C. K., et al. (2024). Using machine learning to predict convection-allowing ensemble forecast skill: Evaluation with the NSSL Warn-on-Forecast System. *Artificial Intelligence for the Earth Systems*, 3, e230106. https://doi.org/10.1175/AIES-D-23-0106.1
[2] Rodwell, M. J., et al. (2013). Characteristics of occasional poor medium-range weather forecasts for Europe. *Bulletin of the American Meteorological Society*, 94(9), 1393–1405. https://doi.org/10.1175/BAMS-D-12-00099.1
[3] Parsons, D. B., et al. (2019). The role of continental mesoscale convective systems in forecast busts within global weather prediction systems. *Atmosphere*, 10(11), 681. https://doi.org/10.3390/atmos10110681
[4] Guan, H., et al. (2022). GEFSv12 reforecast dataset for supporting subseasonal and hydrometeorological applications. *Monthly Weather Review*, 150(3), 647–665. https://doi.org/10.1175/MWR-D-21-0245.1
[5] Hamill, T. M., et al. (2022). The reanalysis for the Global Ensemble Forecast System, version 12. *Monthly Weather Review*, 150(1), 59–79. https://doi.org/10.1175/MWR-D-21-0023.1
[6] Dube, A., Ashrit, R., Singh, H., Iyengar, G., & Rajagopal, E. N. (2016). Verification of medium range probabilistic rainfall forecasts over India. *Pure and Applied Geophysics*, 173, 2489–2510. https://doi.org/10.1007/s00024-016-1278-z
[7] Chakraborty, P., Sarkar, A., Bhatla, R., & Singh, R. (2021). Assessing the skill of NCMRWF global ensemble prediction system in predicting Indian summer monsoon during 2018. *Atmospheric Research*, 251, 105255. https://doi.org/10.1016/j.atmosres.2020.105255
[8] Dube, A., Karunasagar, S., Ashrit, R., & Mitra, A. K. (2022). Spatial verification of ensemble rainfall forecasts over India. *Atmospheric Research*, 273, 106169. https://doi.org/10.1016/j.atmosres.2022.106169
[9] Roy Bhowmik, S. K., & Durai, V. R. (2012). Development of multimodel ensemble based district level medium range rainfall forecast system for Indian region. *Journal of Earth System Science*, 121, 273–285. https://doi.org/10.1007/s12040-012-0160-1
[10] Rasp, S., & Lerch, S. (2018). Neural networks for postprocessing ensemble weather forecasts. *Monthly Weather Review*, 146(11), 3885–3900. https://doi.org/10.1175/MWR-D-18-0187.1
[11] Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359–378. https://doi.org/10.1198/016214506000001437
[12] Br&ouml;cker, J., & Smith, L. A. (2007). Increasing the reliability of reliability diagrams. *Weather and Forecasting*, 22(3), 651–661. https://doi.org/10.1175/WAF993.1
[13] Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems* 30, 4765–4774. https://proceedings.neurips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions
[14] Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2, 56–67. https://doi.org/10.1038/s42256-019-0138-9
[15] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD Conference*, 785–794. https://doi.org/10.1145/2939672.2939785
[16] Dutta, S., & Pal, S. K. (2022). Interpretation of black box for short-term predictions of pre-monsoon cumulonimbus cloud events over Kolkata. *SN Computer Science*, 3, 352. https://doi.org/10.1007/s42979-022-01255-x
[17] Silva, S. J., Keller, C. A., & Hardin, J. (2022). Using an explainable machine learning approach to characterize Earth system model errors: Application of SHAP analysis to modeling lightning flash occurrence. *Journal of Advances in Modeling Earth Systems*, 14, e2021MS002881. https://doi.org/10.1029/2021MS002881
[18] Daif, N., Di Nunno, F., Granata, F., et al. (2025). Forecasting maximal and minimal air temperatures using explainable machine learning: SHAP versus LIME. *Stochastic Environmental Research and Risk Assessment*, 39, 2551–2581. https://doi.org/10.1007/s00477-025-02984-4
[19] Price, I., et al. (2025). Probabilistic weather forecasting with machine learning. *Nature*, 637, 84–90. https://doi.org/10.1038/s41586-024-08252-9
[20] Hersbach, H., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999–2049. https://doi.org/10.1002/qj.3803
[21] Kiladis, G. N., Dias, J., Straub, K. H., Wheeler, M. C., Tulich, S. N., Kikuchi, K., Weickmann, K. M., & Ventrice, M. J. (2014). A comparison of OLR and circulation-based indices for tracking the MJO. *Monthly Weather Review*, 142(5), 1697–1715. https://doi.org/10.1175/MWR-D-13-00301.1
[22] Jarvis, A., Reuter, H. I., Nelson, A., & Guevara, E. (2008). Hole-filled seamless SRTM data V4. International Centre for Tropical Agriculture (CIAT). https://srtm.csi.cgiar.org
[23] NASA Langley Research Center (2024). Prediction Of Worldwide Energy Resources (POWER) project, daily point API. https://power.larc.nasa.gov/api/temporal/daily/point
[24] Global Administrative Areas (GADM) (2022). GADM database of Global Administrative Areas, version 4.1. https://gadm.org/
[25] Copernicus Climate Change Service (C3S) (2023). ERA5 hourly data on single levels from 1940 to present. Copernicus Climate Change Service Climate Data Store. https://cds.climate.copernicus.eu/datasets
[26] National Oceanic and Atmospheric Administration (NOAA). GEFSv12 retrospective forecast archive (reforecast), AWS Registry of Open Data. https://registry.opendata.aws/noaa-gefs-reforecast/
