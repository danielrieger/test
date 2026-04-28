import os
import sys
import traceback
import io
from contextlib import redirect_stdout, redirect_stderr
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("Python Interpreter")

@mcp.tool()
def execute_python(code: str) -> str:
    """
    Executes an arbitrary Python code block and returns the output (stdout/stderr).
    Use this to perform data analysis, run simulations, or interact with project libraries.
    """
    stdout = io.StringIO()
    stderr = io.StringIO()
    
    # 1. Configure Python Path
    thesis_src = "/home/daniel/Thesis/smlm_score/src"
    if thesis_src not in sys.path:
        sys.path.insert(0, thesis_src)
        
    # 2. Configure CUDA Environment Paths
    # We need to explicitly point to the conda-managed CUDA libraries for Numba
    conda_env = "/home/daniel/miniforge3/envs/smlm"
    os.environ["CUDA_HOME"] = conda_env
    
    # List of paths where CUDA/NVVM libraries might live in this env
    cuda_libs = [
        f"{conda_env}/lib",
        f"{conda_env}/nvvm/lib64"
    ]
    
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    for lib in cuda_libs:
        if lib not in current_ld:
            current_ld = f"{lib}:{current_ld}" if current_ld else lib
    os.environ["LD_LIBRARY_PATH"] = current_ld

    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            # Execute the code
            exec(code, globals())
        
        result = stdout.getvalue()
        errors = stderr.getvalue()
        
        final_output = result
        if errors:
            final_output += f"\n--- Standard Error ---\n{errors}"
        
        return final_output if final_output else "Execution successful (no output)."
        
    except Exception:
        return f"Error during execution:\n{traceback.format_exc()}"

if __name__ == "__main__":
    mcp.run()
