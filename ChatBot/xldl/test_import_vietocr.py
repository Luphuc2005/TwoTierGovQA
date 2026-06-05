import sys
import traceback

print("Testing vietocr import...")
try:
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg
    print("VictOCR imported successfully!")
except Exception as e:
    print("Error during import:")
    traceback.print_exc()
