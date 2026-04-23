---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: epoxy-diffusion
    language: python
    name: python3
---

```python
from ml_tools.data_exploration import (summarize_dataframe,
                                       drop_macro,
                                       clean_column_names,
                                       split_continuous_categorical_targets,
                                       plot_value_distributions,
                                       plot_numeric_overview_boxplot_macro,
                                       plot_continuous_vs_target,
                                       plot_categorical_vs_target,
                                       plot_correlation_heatmap,
                                       encode_categorical_features,
                                       finalize_feature_schema,
                                       filter_subset_categorical,
                                       show_null_columns)
from ml_tools.outlier_detection import (isolation_forest,
                                        local_outlier_factor,
                                        drop_outliers_mask,
                                        clip_outliers_multi)
from ml_tools.utilities import load_dataframe, save_dataframe_with_schema, merge_dataframes
from ml_tools.schema import FeatureSchema

from paths import PM
from helpers.constants import (TARGET, 
                               # categorical features
                               EPOXY, 
                               CURING, 
                               FILLER,                                
                               # unused targets
                               FLEXURAL_STRENGTH, 
                               ELONGATION_AT_BREAK, 
                               IMPACT_STRENGTH)
```

## 1. Load Data

```python
df_raw, _ = load_dataframe(df_path=PM.preprocessed_file, kind="pandas")
```

```python
# drop unused targets
df_raw = df_raw.drop(columns=[FLEXURAL_STRENGTH, ELONGATION_AT_BREAK, IMPACT_STRENGTH])
```

```python
# Select Epoxy E-51
df_raw_e51 = filter_subset_categorical(df=df_raw,
                                       filters={EPOXY: "E-51"},
                                       drop_filter_cols=True,
                                       reset_index=True)
```

## 2. Clean Data

```python
df_cleanI = drop_macro(df=df_raw_e51, 
                       log_directory=PM.engineering, 
                       targets=[TARGET],
                       skip_targets=True,
                       threshold=0.7)
```

```python
df_cleanII = clean_column_names(df=df_cleanI)
```

```python
summarize_dataframe(df_cleanII)
```

```python
from helpers.constants import EPOXY_CURING_RATIO, FILLER_PROPORTION, TEMPERATURE

ALLOWED_RANGES = {
    EPOXY_CURING_RATIO: (0,10),
    FILLER_PROPORTION: (0,30),
    TEMPERATURE: (295,450),
    TARGET: (1, 100)
}

df_cleanIII = clip_outliers_multi(df=df_cleanII, clip_dict=ALLOWED_RANGES)
```

```python
summarize_dataframe(df_cleanIII)
```

## 3. Outlier Detection

```python
df_clean = df_cleanIII
```

```python
outliers_mask_forest = isolation_forest(df_features=df_clean,
                                        ignore_columns=[TARGET],
                                        shadow_median_imputation=False)
```

```python
outliers_mask_lof = local_outlier_factor(df_features=df_clean,
                                        ignore_columns=[TARGET],
                                        shadow_median_imputation=False)
```

```python
df_clean[outliers_mask_lof & outliers_mask_forest]
```

```python
df_processed = drop_outliers_mask(df=df_clean,
                              outlier_mask=outliers_mask_forest & outliers_mask_lof,
                              reset_index=True)
```

```python
summarize_dataframe(df_processed)
```

## 4. Split Features and Targets

```python
df_continuous, df_categorical, df_targets = split_continuous_categorical_targets(df=df_processed,
                                                                                categorical_cols=[CURING, FILLER],
                                                                                target_cols=[TARGET])
```

```python
show_null_columns(df=df_continuous)
```

```python
show_null_columns(df=df_categorical)
```

```python
show_null_columns(df=df_targets)
```

## 5. Plots

```python
plot_value_distributions(df=df_processed, save_dir=PM.engineering_plots)
```

```python
plot_correlation_heatmap(df=df_processed, plot_title="Data Correlation Heatmap", save_dir=PM.engineering_plots)
```

```python
plot_numeric_overview_boxplot_macro(df=df_continuous,
                                    save_dir=PM.engineering_plots,
                                    plot_title="Continuous Feature Distributions",
                                    handle_zero_variance="constant")
```

```python
plot_continuous_vs_target(df_continuous=df_continuous, df_targets=df_targets, save_dir=PM.engineering_plots)
```

```python
plot_categorical_vs_target(df_categorical=df_categorical, df_targets=df_targets, save_dir=PM.engineering_plots)
```

## 6. Encode Categorical Features

```python
df_categorical_encoded, categorical_mapping = encode_categorical_features(df_categorical=df_categorical,
                                                                          encode_nulls=True,
                                                                          null_label="Other")
```

## 7. Reconstruct DataFrame

```python
df_features_final = merge_dataframes(df_continuous, df_categorical_encoded)
```

```python
df_final = merge_dataframes(df_features_final, df_targets, reset_index=True)
```

## 8. Create Feature Schema

```python
schema = finalize_feature_schema(df_features=df_features_final, categorical_mappings=categorical_mapping)
```

## 9. Save Artifacts

```python
save_dataframe_with_schema(df=df_final, full_path=PM.engineering_file, schema=schema)
```

```python
schema.to_json(PM.engineering)
schema.save_artifacts(PM.engineering)
```
