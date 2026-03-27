import sys

# Test if globals().update() updates module attributes

def update_global():
    globals().update({"test_value": "updated"})

test_value = "original"
print(f"Before: test_value={test_value}")
print(f"Before (via module): {sys.modules[__name__].test_value}")

update_global()

print(f"After: test_value={test_value}")
print(f"After (via module): {sys.modules[__name__].test_value}")

# Also test via import
import __main__ as main_mod
print(f"Via import: {main_mod.test_value}")
