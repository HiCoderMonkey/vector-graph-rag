import hashlib
import os

def check_model_update(model_dir, model_name):
    def calculate_md5(file_path):
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    md5_name = "model_md5_value.txt"
    md5_file_path = os.path.join(model_dir, md5_name)
    model_path = os.path.join(model_dir, model_name)
    if not os.path.exists(md5_file_path):
        md5_value = calculate_md5(model_path)
        with open(md5_file_path, 'w') as f:
            f.write(md5_value)
        return True
    else:
        with open(md5_file_path, 'r') as f:
            old_md5 = f.read()
        md5_value = calculate_md5(model_path)
        if old_md5 != md5_value:
            with open(md5_file_path, 'w') as f:
                f.write(md5_value)
            return True
        else:
            return False

    with open(md5_name, 'w') as f:
        f.write(md5_value)
