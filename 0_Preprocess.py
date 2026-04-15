from helpers.function_map import TRANSFORMATION_RECIPE
from paths import PM
from ml_tools.ETL_engineering import DragonProcessor


if __name__ == "__main__":
    # instantiate processor
    processor = DragonProcessor(recipe=TRANSFORMATION_RECIPE)
    
    # Process df
    processor.load_transform_save(input_path=PM.clean_file, output_path=PM.preprocessed_file)
