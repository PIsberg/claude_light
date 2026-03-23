import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from claude_light import _resolve_new_content

original_code = """\
def my_func():
    try:
        import foo
        # start doing stuff
        a = 1
        b = 2
    except:
        pass
"""

# The LLM forgot the explicit 8-space indentation!
llm_search = """\
import foo
# start doing stuff
a = 1
b = 2
"""

llm_replace = """\
import foo
import warnings
warnings.filterwarnings("ignore")
a = 1
b = 2
"""

print("Running test...")
try:
    result = _resolve_new_content("demo.py", original_code, llm_search, llm_replace)
    print("SUCCESS")
    print("--- RESULT ---")
    print(result)
except Exception as e:
    print(f"FAILED: {e}")
