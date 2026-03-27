import os
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-dummy-test-key"

import claude_light as cl
print(f"Initial skeleton_context: {repr(cl.skeleton_context[:50])}")
print(f"Initial state.skeleton_context: {repr(cl.state.skeleton_context[:50])}")

cl._apply_skeleton("new skeleton text")

print(f"After _apply_skeleton: {repr(cl.skeleton_context[:50])}")
print(f"After _apply_skeleton state: {repr(cl.state.skeleton_context[:50])}")
