import os

def rename_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".tif") and filename.endswith("_x4.tif"):
            new_filename = filename.replace("_x4.tif", ".tif")
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_filename)
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {filename} -> {new_filename}")

folder_path = "/data/55d_1/IRR-DiffSR/data_samples/My_UCMerced_LandUse_test_bicubic_x4_all"
rename_images(folder_path)
