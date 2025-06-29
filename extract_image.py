import os
import random
import shutil

def copy_random_images(src_folder, dst_folder, num_images):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    all_images = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]

    selected_images = random.sample(all_images, min(num_images, len(all_images)))

    for image in selected_images:
        src_path = os.path.join(src_folder, image)
        dst_path = os.path.join(dst_folder, image)
        shutil.copy(src_path, dst_path)

    print(f"Successfully copy {len(selected_images)} images to {dst_folder}")

# 'ADM', 'BigGAN', 'glide', 'Midjourney', 'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'VQDM', 'wukong'
# [airplane,bird,bicycle,boat,bottle,bus,car,cat,cow,chair,diningtable,dog,person,pottedplant,motorbike,tvmonitor,train,sheep,sofa,horse]

classes = ['glide_50_27']
num = 1600 # number of images extracted

for cls in classes:
    source_folder = f'G:/Dataset/all_test/{cls}'  # src
    destination_folder = f'G:/Dataset/SNE/{cls}'  # dst

    source_real_path = os.path.join(source_folder, '0_real')
    destination_real_path = os.path.join(destination_folder, '0_real')

    source_fake_path = os.path.join(source_folder, '1_fake')
    destination_fake_path = os.path.join(destination_folder, '1_fake')

    copy_random_images(source_real_path, destination_real_path, num_images=num//2)
    copy_random_images(source_fake_path, destination_fake_path, num_images=num//2)