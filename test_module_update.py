
# test_module_update.py
test_var = "old"

def update_var():
    globals()["test_var"] = "new"

# Test
print(f"Before: {test_var}")
update_var()
print(f"After: {test_var}")

# Also test with import
import test_module_update as tm
print(f"Imported Before: {tm.test_var}")
tm.update_var()
print(f"Imported After: {tm.test_var}")
