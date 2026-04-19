import os
import sys
import json
import subprocess
import random
import time
from pathlib import Path

# Fix python path for subprocesses
project_root = str(Path(__file__).parent.parent.absolute())
env = os.environ.copy()
if "PYTHONPATH" in env:
    env["PYTHONPATH"] = f"{project_root};{env['PYTHONPATH']}"
else:
    env["PYTHONPATH"] = project_root

CONFIG_PATH = os.path.join(project_root, "examples", "pipeline_config.json")
SCRIPT_PATH = os.path.join(project_root, "examples", "NPC_example_BD.py")

SCORING_TYPES = ["Tree", "GMM", "Distance"]
MODES = ["bayesian", "frequentist", "brownian"]

def run_test(iteration):
    print(f"\n{'='*50}")
    print(f"--- RUN {iteration}/10 ---")
    
    # 1. Read existing config
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
        
    # 2. Randomize parameters
    score = random.choice(SCORING_TYPES)
    mode = random.choice(MODES)
    pct = random.randint(5, 20) # 5% to 20% spatial slice
    
    print(f"Randomized Config: Mode={mode}, Scoring={score}, SpatialSlice={pct}%")
    
    # Update config
    config["filtering"]["type"] = "random"
    if "random" not in config["filtering"]:
        config["filtering"]["random"] = {}
    config["filtering"]["random"]["size_percentage"] = pct
    
    config["optimization"]["mode"] = mode
    config["optimization"]["bayesian"]["scoring_type"] = score
    config["optimization"]["bayesian"]["number_of_frames"] = 2  # Keep it short for testing
    config["optimization"]["frequentist"]["scoring_type"] = score
    config["optimization"]["brownian"]["scoring_type"] = score
    config["optimization"]["brownian"]["number_of_frames"] = 10 # Keep it short
    
    # Force min cluster size low to ensure it finds *something* in tiny 5% slices
    config["clustering"]["min_cluster_size"] = 10
    config["clustering"]["min_npc_points"] = 40
    
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=4)
        
    # 3. Run subprocess
    start_t = time.time()
    result = subprocess.run([sys.executable, SCRIPT_PATH], capture_output=True, text=True, env=env)
    elapsed = time.time() - start_t
    
    # 4. Analyze output for unplausible data
    output = result.stdout + "\n" + result.stderr
    
    errors = []
    if result.returncode != 0:
        errors.append(f"Process crashed with exit code {result.returncode}")
        
    if "Traceback" in output:
        errors.append("Traceback detected in output")
    if "ValueError" in output:
        errors.append("ValueError detected in output")
    if "MemoryError" in output:
        errors.append("MemoryError detected in output")
        
    if mode == "frequentist":
        if "Improvement:   0.0000" in output:
            errors.append("Frequentist improvement was exactly 0.0000 (Optimization failed)")
            
    if mode == "bayesian":
        # Check if score is just 0.0 repeatedly
        if "score: 0.0" in output.lower():
            errors.append("Bayesian scores equal to 0.0 detected")
            
    if not errors:
        print(f"✅ SUCCESS: Run {iteration} completed cleanly in {elapsed:.1f}s.")
        return True, output
    else:
        print(f"❌ FAILED: Run {iteration} encountered issues in {elapsed:.1f}s:")
        for e in errors:
            print(f"  - {e}")
            
        log_file = f"failed_run_{iteration}.log"
        with open(log_file, "w") as f:
            f.write(output)
        print(f"  Full log written to {log_file}")
        return False, output

def main():
    print("Starting Automated Random Pipeline Tester")
    print("Will execute 10 random permutations of the SMLM-IMP pipeline.")
    
    success_count = 0
    for i in range(1, 11):
        success, _ = run_test(i)
        if success:
            success_count += 1
            
    print(f"\n{'='*50}")
    print(f"TESTING COMPLETE.")
    print(f"Passed: {success_count}/10")
    if success_count < 10:
        print("Please review the generated failed_run_*.log files for details.")

if __name__ == "__main__":
    main()
