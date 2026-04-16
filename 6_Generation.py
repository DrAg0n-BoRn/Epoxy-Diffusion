from ml_tools.ML_inference_diffusion import DragonDiTGuidedGenerator
from ml_tools.ML_models_diffusion import DragonDiTGuided, DragonAutoencoder
from ml_tools.ML_utilities import DragonArtifactFinder
from ml_tools.serde import deserialize_object

from paths import PM
from helpers.constants import TARGET_RANGE


BATCH_SIZE = None # Load default from file, or override with an integer value.
GUIDANCE_SCALE = 3.0


def main():
    _USED_BATCH_SIZE = BATCH_SIZE if type(BATCH_SIZE) is int else deserialize_object(PM.batch_size_file, expected_type=int)
    
    finder_autoencoder = DragonArtifactFinder(directory=PM.autoencoder, load_scaler=True, load_schema=False, strict=True)
    autoencoder = DragonAutoencoder.from_artifact_finder(finder_autoencoder)
    
    finder_diffusion = DragonArtifactFinder(directory=PM.diffusion, load_scaler=True, load_schema=False, strict=True)
    diffusion_model = DragonDiTGuided.from_artifact_finder(finder_diffusion)
    
    generator = DragonDiTGuidedGenerator(save_dir=PM.generation,
                                         diffusion_model=diffusion_model,
                                         encoder=autoencoder,
                                         device="cuda:0")
    
    generator.generate_plot_multi(targets=list(range(TARGET_RANGE[0], TARGET_RANGE[1]+1, 10)),
                                  batch_size=_USED_BATCH_SIZE,
                                  guidance_scale=GUIDANCE_SCALE,
                                  ode_steps=20,
                                  positive_columns="all",
                                  round_float_columns="all",
                                  float_rounding_precision=2,
                                  handle_zero_variance="constant")


if __name__ == "__main__":
    main()
