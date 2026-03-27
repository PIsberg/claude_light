import os
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-dummy-test-key"

from unittest.mock import patch, MagicMock

import claude_light.llm as llm
print(f"llm.client type: {type(llm.client)}")
print(f"llm.client: {llm.client}")
print(f"hasattr messages: {hasattr(llm.client, 'messages')}")
if hasattr(llm.client, 'messages'):
    print(f"llm.client.messages type: {type(llm.client.messages)}")
    print(f"llm.client.messages.create: {getattr(llm.client.messages, 'create', 'NOT FOUND')}")

# Try patching
resp = MagicMock()
resp.content = [MagicMock(type="text", text="test")]
resp.usage = MagicMock(input_tokens=10, output_tokens=5, cache_creation_input_tokens=0, cache_read_input_tokens=0)

with patch.object(llm.client.messages, 'create', return_value=resp) as mock:
    print(f"After patch: {llm.client.messages.create}")
    result = llm.client.messages.create()
    print(f"Mock called: {mock.called}")
    print(f"Result: {result}")
