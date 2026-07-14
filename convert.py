import ctranslate2

model_path = "marianmt-mtet-model-exp9/final_model"
output_path = "marianmt-mtet-model-exp9/final_model_ct2"

print(f"Converting {model_path} to {output_path}...")
try:
    converter = ctranslate2.converters.TransformersConverter(model_path)
    converter.convert(output_path)
    print("Conversion successful.")
except Exception as e:
    print(f"Conversion failed: {e}")
