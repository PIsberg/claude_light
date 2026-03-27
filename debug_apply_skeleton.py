import os
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-dummy-test-key"

import claude_light as cl

# Monkey patch to debug
original_sync = cl._sync_bindings
original_refresh = cl._refresh_exports
orig_apply = cl.llm._apply_skeleton

call_count = [0]

def debug_sync():
    call_count[0] += 1
    print(f"[{call_count[0]}] _sync_bindings(): before: module.skeleton_context={repr(cl.skeleton_context[:20])} state.skeleton_context={repr(cl.state.skeleton_context[:20])}")
    result = original_sync()
    print(f"[{call_count[0]}] _sync_bindings(): after: module.skeleton_context={repr(cl.skeleton_context[:20])} state.skeleton_context={repr(cl.state.skeleton_context[:20])}")
    return result

def debug_refresh():
    call_count[0] += 1
    print(f"[{call_count[0]}] _refresh_exports(): before: module.skeleton_context={repr(cl.skeleton_context[:20])} state.skeleton_context={repr(cl.state.skeleton_context[:20])}")
    result = original_refresh()
    print(f"[{call_count[0]}] _refresh_exports(): after: module.skeleton_context={repr(cl.skeleton_context[:20])} state.skeleton_context={repr(cl.state.skeleton_context[:20])}")
    return result

def debug_apply(skeleton):
    call_count[0] += 1
    print(f"[{call_count[0]}] llm._apply_skeleton({repr(skeleton[:20])}): before: state.skeleton_context={repr(cl.state.skeleton_context[:20])}")
    result = orig_apply(skeleton)
    print(f"[{call_count[0]}] llm._apply_skeleton(): after: state.skeleton_context={repr(cl.state.skeleton_context[:20])}")
    return result

cl._sync_bindings = debug_sync
cl._refresh_exports = debug_refresh
cl.llm._apply_skeleton = debug_apply

print("\n=== CALLING _apply_skeleton ===\n")
cl._apply_skeleton("new skeleton text")
print("\n=== RESULT ===")
print(f"module.skeleton_context={repr(cl.skeleton_context)}")
print(f"state.skeleton_context={repr(cl.state.skeleton_context)}")
