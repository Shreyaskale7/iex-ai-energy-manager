import shutil
import zipfile
import os

directories = ["src", "scripts", "models", "data", "forecasts", "demo", "reports"]
files = ["HANDOFF_README.md", "pyproject.toml", "requirements.txt"]

with zipfile.ZipFile("iex_forecast_handoff.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
    for d in directories:
        if os.path.exists(d):
            for root, dirs, filenames in os.walk(d):
                for file in filenames:
                    file_path = os.path.join(root, file)
                    try:
                        zipf.write(file_path, os.path.relpath(file_path, "."))
                    except Exception as e:
                        print(f"Skipping {file_path} due to error: {e}")
    for f in files:
        if os.path.exists(f):
            try:
                zipf.write(f, f)
            except Exception as e:
                print(f"Skipping {f} due to error: {e}")

print("Zip created successfully.")
