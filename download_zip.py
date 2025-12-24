import shutil

dir_name = "qwen_lora_weights"
output_filename = dir_name

shutil.make_archive(output_filename, 'zip', dir_name)
print(f"Successfully zipped {dir_name} into {output_filename}.zip")