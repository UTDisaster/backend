import os

TILES_PATH  = r"C:\Users\furkm\OneDrive\Desktop\database_setup\CS4485_Hurricane_Grid_Final\CS4485_Hurricane_Grid_Final"
HOUSES_PATH = r"C:\Users\furkm\OneDrive\Desktop\database_setup\CS4485_Hurricane_Org_final\CS4485_Hurricane_Org_final"

print("=== TILES folder contents ===")
for f in os.listdir(TILES_PATH):
    full = os.path.join(TILES_PATH, f)
    json_path = os.path.join(full, "master_grid_labels.json")
    print(f"  {f} → json exists: {os.path.exists(json_path)}")

print("\n=== HOUSES folder contents ===")
for f in os.listdir(HOUSES_PATH):
    full = os.path.join(HOUSES_PATH, f)
    json_path = os.path.join(full, "labels.json")
    print(f"  {f} → json exists: {os.path.exists(json_path)}")