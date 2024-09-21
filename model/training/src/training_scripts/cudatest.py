# ==============================================================================
# CODEVERITUS SYSTEM CORE MODULE
# ==============================================================================
# File: model/training/src/training_scripts/cudatest.py
# Author: ABHIMANYU993
# Email: abhimanyubadiger1001@gmail.com
# Project: Codeveritus - AI vs Human Code Classifier
# Description: Modular component containing system configuration or script details.
# ==============================================================================

import torch

print(torch.cuda.is_available())

if torch.cuda.is_available():
    print("Cuda is Availabe")
else:
    print("Cuda Can't be found")

# fix device placement bug in eval loop

# update batch normalization momentum

# reformat code to pep8 standards

# adjust cross entropy loss weights