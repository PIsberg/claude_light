import re

with open('tests/test_claude_light.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace patch paths
content = content.replace(
    'patch("claude_light.client.messages.create"',
    'patch("claude_light.llm.client.messages.create"'
)

with open('tests/test_claude_light.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Replaced all patch paths")
