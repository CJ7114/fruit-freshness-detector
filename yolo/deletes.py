import os

def delete_extra_images(base_path, max_images=200):
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}  # Adjust if needed
    
    for category in ['fresh', 'rotten']:
        category_path = os.path.join(base_path, category)
        if not os.path.exists(category_path):
            continue
        
        for subfolder in os.listdir(category_path):
            subfolder_path = os.path.join(category_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            
            images = [img for img in os.listdir(subfolder_path) if os.path.splitext(img)[1].lower() in allowed_extensions]
            images.sort(key=lambda x: os.path.getctime(os.path.join(subfolder_path, x)))  # Sort by creation time
            
            if len(images) > max_images:
                images_to_delete = images[max_images:]  # Keep the first `max_images`, delete the rest
                
                for img in images_to_delete:
                    img_path = os.path.join(subfolder_path, img)
                    try:
                        os.remove(img_path)
                        print(f"Deleted: {img_path}")
                    except Exception as e:
                        print(f"Error deleting {img_path}: {e}")

if __name__ == "__main__":
    dataset_path = "validation"  # Update this path if necessary
    delete_extra_images(dataset_path, max_images=200)
