import os
import shutil

# Define your paths
BASE_DIR = r"D:\AI\Hand_symbol\word_dataset"
# If your images are inside 'images/train' or 'images/test', point to those
SOURCE_FOLDERS = [
    os.path.join(BASE_DIR, "images", "train"),
    os.path.join(BASE_DIR, "images", "test")
]

def reorganize():
    count = 0
    for src in SOURCE_FOLDERS:
        if not os.path.exists(src):
            print(f"Skipping missing folder: {src}")
            continue

        print(f"Processing: {src}")
        for filename in os.listdir(src):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Filenames look like 'bathroom.a06d...jpg'
                # We split by the first dot to get the label
                label = filename.split('.')[0]
                
                # Create the target label directory
                target_dir = os.path.join(BASE_DIR, label)
                os.makedirs(target_dir, exist_ok=True)
                
                # Move the file
                src_path = os.path.join(src, filename)
                dst_path = os.path.join(target_dir, filename)
                
                try:
                    shutil.move(src_path, dst_path)
                    count += 1
                except Exception as e:
                    print(f"Error moving {filename}: {e}")

    print(f"\nDone! Moved {count} images into labeled folders.")
    print(f"Your dataset is now at: {BASE_DIR}")

if __name__ == "__main__":
    reorganize()