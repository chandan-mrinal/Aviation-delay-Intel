import os
import shutil

def prepare_hf_deployment():
    """
    Packages all necessary files into a clean 'hf_app/' directory 
    ready for HuggingFace Spaces deployment.
    """
    target_dir = "hf_app"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    
    # 1. Directories to copy
    dirs_to_copy = ['src', 'results', 'data', '.streamlit']
    for d in dirs_to_copy:
        if os.path.exists(d):
            shutil.copytree(d, os.path.join(target_dir, d))
            
    # 2. Files to copy
    files_to_copy = ['requirements.txt', 'dashboard/app.py', 'README.md', 'findings.md']
    for f in files_to_copy:
        if os.path.exists(f):
            # If it's the app.py, place it in the root
            if f == 'dashboard/app.py':
                shutil.copy(f, os.path.join(target_dir, 'app.py'))
            else:
                shutil.copy(f, os.path.join(target_dir, f))
                
    print(f"--- Deployment Package Created in '{target_dir}/' ---")
    print("Next Steps:")
    print("1. Create a New Streamlit Space on HuggingFace.")
    print(f"2. Upload all files from '{target_dir}/' to the Space.")
    print("3. Set your GOOGLE_API_KEY and CHECKWX_API_KEY in the Space Secrets.")

if __name__ == "__main__":
    prepare_hf_deployment()
