import os

root_folder = r"C:\AA_Desktop\VsCode\Sentry_Turret_V1\assets\faces\raw_images"

names = [
    name for name in os.listdir(root_folder)
    if os.path.isdir(os.path.join(root_folder, name))
]

print(names)