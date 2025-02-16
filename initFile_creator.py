import os

# Define the package directories where __init__.py should be created
package_dirs = [
    "src/particle_analysis",
    "src/particle_analysis/core",
    "src/particle_analysis/analysis",
    "src/particle_analysis/io",
    "src/particle_analysis/visualization",
    "src/particle_analysis/gui",
    "tests"
]

# Define package module mappings
package_modules = {
    "src/particle_analysis/core": ["particle_detection", "particle_tracking", "feature_calculation"],
    "src/particle_analysis/analysis": ["diffusion", "statistics"],
    "src/particle_analysis/io": ["readers", "writers"],
    "src/particle_analysis/visualization": ["plot_utils", "viewers"],
    "src/particle_analysis/gui": ["main_window", "analysis_widget", "viewers", "batch_dialog", "settings_dialog", "results_viewer", "analysis_dashboard"],
    "tests": ["test_particle_detection", "test_particle_tracking", "test_feature_calculation", "test_diffusion", "test_io"]
}

# Create __init__.py files
for package in package_dirs:
    init_path = os.path.join(package, "__init__.py")
    modules = package_modules.get(package, [])

    with open(init_path, "w") as f:
        f.write(f"\"\"\"\nInitialization file for {package.replace('/', '.')}\n\"\"\"\n")
        if modules:
            f.write("__all__ = [\"" + "\", \"".join(modules) + "\"]\n")

print("All __init__.py files have been generated.")
