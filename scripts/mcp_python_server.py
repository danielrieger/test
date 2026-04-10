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
    
    # Pre-configure environment (e.g., PYTHONPATH)
    # This ensures that the 'smlm_score' package is available
    thesis_root = r"C:\Users\User\OneDrive\Desktop\Thesis"
    if thesis_root not in sys.path:
        sys.path.insert(0, thesis_root)
        
    # Ensure Library/bin is in PATH for DLL resolution (IMP, etc.)
    env_root = r"C:\envs\py311"
    lib_bin = os.path.join(env_root, "Library", "bin")
    if lib_bin not in os.environ["PATH"]:
        os.environ["PATH"] = f"{lib_bin};{os.environ['PATH']}"

    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            # Execute the code in the context of the current global dictionary
            # For persistent state, we could use a custom dict, but here we just run it.
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
