# pipelines/compile_pipeline.py
import kfp
from kfp.v2 import compiler
from trading_pipeline import btc_trading_pipeline # Assuming your pipeline file is trading_pipeline.py

# Define the output path for the compiled pipeline JSON
# This path can be relative to the execution directory of this script
PIPELINE_JSON_PKG_PATH = "trading_pipeline.json" 
# This could also be an absolute path or configured via an environment variable if needed

if __name__ == "__main__":
    # KFP V2 compiler (Vertex AI Pipelines uses KFP v2 style)
    # For KFP v2, the mode is typically not needed as it defaults to V2_COMPATIBLE or similar
    # if using kfp.v2.compiler.Compiler.
    # If you installed full `kfp` which might include v1 compiler, ensure you use v2.
    
    # Ensure you are using kfp.v2.compiler.Compiler for Vertex AI
    # from kfp.v2 import compiler as compiler_v2 # If you need to be explicit
    
    compiler.Compiler().compile(
        pipeline_func=btc_trading_pipeline,
        package_path=PIPELINE_JSON_PKG_PATH,
        # type_check=True # Optional: for stricter type checking during compilation
    )
    print(f"KFP pipeline compiled successfully to: {PIPELINE_JSON_PKG_PATH}")
