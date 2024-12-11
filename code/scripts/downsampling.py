from PIL import Image
import os

def downsample_images(hr_folder_path, scale_factor, lr_folder_path, resample_method='bicubic'):
    os.makedirs(lr_folder_path, exist_ok=True)
    hr_files = os.listdir(hr_folder_path)

    # 选择下采样方法
    resample_dict = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'lanczos': Image.LANCZOS
    }
    resample = resample_dict.get(resample_method.lower(), Image.BICUBIC)
    
    for hr_file in hr_files:
        hr_file_path = os.path.join(hr_folder_path, hr_file)
        if os.path.isfile(hr_file_path) and hr_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            hr_image = Image.open(hr_file_path)
            hr_width, hr_height = hr_image.size
            lr_width = hr_width // scale_factor
            lr_height = hr_height // scale_factor
            lr_image = hr_image.resize((lr_width, lr_height), resample=resample)
            lr_file_path = os.path.join(lr_folder_path, hr_file)
            lr_image.save(lr_file_path)
            print(f"LR saving to {lr_file_path}")

if __name__ == "__main__":
    hr_image_path = "/data/55d_1/IRR-DiffSR/data_samples/UFO-120/TEST/lrd"
    scale_factor = 2
    lr_image_path = "/data/55d_1/IRR-DiffSR/data_samples/UFO-120/TEST/lr_x4"
    resample_method = "bicubic"
    
    downsample_images(hr_image_path, scale_factor, lr_image_path, resample_method)
