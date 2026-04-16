# Raw targets
'''
    "断裂韧性": "Fracture Toughness(MPa m0.5)", # Missing 91% of data ⛔
    "弯曲强度": "Flexural Strength(MPa)",   # Missing 51% of data ⚠️
    "弯曲模量": "Flexural Modulus(MPa)",    # Missing 74% of data ⛔
    "冲击强度": "Impact Strength(kJ/m2)",   # Missing 64% of data ⛔️
    "杨氏模量": "Young Modulus(MPa)",   # Missing 69% of data ⛔
    "拉伸强度": "Tensile Strength(MPa)",    # Missing 23% of data ✅
    "剪切强度": "Shear Strength(MPa)", # Missing 94% of data ⛔
    "断裂伸长率": "Elongation at Break(%)"  # Missing 63% of data ⛔
'''

TARGET = "Tensile Strength(MPa)"
# Secondary targets (currently ignored due to high missingness)
FLEXURAL_STRENGTH = "Flexural Strength(MPa)"
IMPACT_STRENGTH = "Impact Strength(kJ/m2)"
ELONGATION_AT_BREAK = "Elongation at Break(%)"


# Base Features
EPOXY = "Epoxy" # Use epoxy E-51 only then drop this feature
CURING = "Curing" # categorical
FILLER = "Filler" # categorical 
EPOXY_EPOXY_RATIO = "Epoxy/Epoxy Ratio" # continuous
EPOXY_CURING_RATIO = "Epoxy/Curing Ratio" # continuous
FILLER_PROPORTION = "Filler Proportion(%)" # continuous 
FILLER_PROPORTION_2 = "Filler Proportion 2(%)" # continuous
FILLER_PROPORTION_3 = "Filler Proportion 3(%)" # continuous
TEMPERATURE = "Temperature(K)" # continuous

# Model hyperparameters
EMBEDDING_DIM = 128

# target range for generation
TARGET_RANGE = (60, 80)
