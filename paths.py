from ml_tools.path_manager import DragonPathManager


# 1. Initialize the PathManager using this file as the anchor, adding base directories.
PM = DragonPathManager(
    anchor_file=__file__,
    base_directories=["helpers", "start_data", "results", "backups"]
)

# 2. Define directories and files.
### Base files
PM.clean_file = PM.start_data / "clean_data.csv"

### Datasets
PM.datasets = PM.results / "Datasets"
PM.preprocessed_file = PM.datasets / "preprocessed_data.csv"

### Feature Engineering
PM.engineering = PM.results / "Feature Engineering"
PM.engineering_plots = PM.engineering / "Engineering Plots"
PM.engineering_file = PM.datasets / "engineered_data.csv"

### MICE - VIF
PM.mice = PM.results / "MICE"
PM.mice_datasets = PM.mice / "MICE Datasets"
PM.vif = PM.results / "VIF"
PM.imputed_file = PM.datasets / "imputed_data.csv"

### Autoencoder
PM.autoencoder = PM.results / "Autoencoder"

### Diffusion
PM.diffusion = PM.results / "Diffusion"

### Generation
PM.generation = PM.results / "Generation"
PM.train_comparison = PM.generation / "Train Range Comparison"
PM.batch_size_file = PM.generation / "batch_size.joblib"


# 3. Make directories and check status
PM.make_dirs()

if __name__ == "__main__":
    PM.status()
