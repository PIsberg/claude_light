import os
import sys

os.environ["ANTHROPIC_API_KEY"] = "sk-ant-dummy-test-key"

import claude_light as cl

print(f"cl._apply_skeleton: {cl._apply_skeleton}")

print(f"\nBefore:")
print(f"  cl.skeleton_context: {repr(cl.skeleton_context[:30] if cl.skeleton_context else 'EMPTY')}")
print(f"  cl.state.skeleton_context: {repr(cl.state.skeleton_context[:30] if cl.state.skeleton_context else 'EMPTY')}")

cl._apply_skeleton("new skeleton text")

print(f"\nAfter:")
print(f"  cl.skeleton_context: {repr(cl.skeleton_context[:30] if cl.skeleton_context else 'EMPTY')}")
print(f"  cl.state.skeleton_context: {repr(cl.state.skeleton_context[:30] if cl.state.skeleton_context else 'EMPTY')}")
print(f"  sys.modules['claude_light'].skeleton_context: {repr(sys.modules['claude_light'].skeleton_context[:30] if sys.modules['claude_light'].skeleton_context else 'EMPTY')}")

# Check if globals were updated
print(f"\nModule __dict__['skeleton_context']: {repr(sys.modules['claude_light'].__dict__['skeleton_context'][:30] if sys.modules['claude_light'].__dict__['skeleton_context'] else 'EMPTY')}")

# Check the actual llm module function
print(f"\nDirect call to llm._apply_skeleton:")
cl.llm._apply_skeleton("direct test")
print(f"  cl.state.skeleton_context: {repr(cl.state.skeleton_context[:30] if cl.state.skeleton_context else 'EMPTY')}")
print(f"  cl.skeleton_context: {repr(cl.skeleton_context[:30] if cl.skeleton_context else 'EMPTY')}")
