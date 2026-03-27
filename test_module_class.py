import sys
import types

class TestModule(types.ModuleType):
    def __setattr__(self, name, value):
        print(f"__setattr__ called: {name}={repr(value)}")
        super().__setattr__(name, value)

# Create a test module
mod = TestModule("test")
sys.modules["test_mod"] = mod
exec("""
test_value = "original"

def update():
    print(f"Inside update function: test_value={test_value}")
    print("Calling globals().update()...")
    globals().update({"test_value": "updated"})
    print(f"After update: test_value={test_value}")
""", mod.__dict__)

sys.modules["test_mod"].__class__ = TestModule

# Now test
print("Before:")
print(f"  test_value={mod.test_value}")

print("\nCalling update():")
mod.update()

print("\nAfter:")
print(f"  test_value={mod.test_value}")
