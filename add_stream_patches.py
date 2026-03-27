import re

with open('tests/test_claude_light.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find lines with patch("claude_light.llm.client.messages.create" and add stream_chat_response patch
new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)
    
    # If this line has a with statement with client.messages.create patch,
    # check if it also has stream_chat_response patch
    if 'patch("claude_light.llm.client.messages.create"' in line and 'stream_chat_response' not in line:
        # Look back to find the with statement start
        j = i
        while j >= 0 and 'with patch' not in lines[j]:
            j -= 1
        
        if j >= 0:
            # Insert stream_chat_response patch on the next line if it's a continuation
            # Check if next line has another patch or if we're at the start of a with block
            if i + 1 < len(lines) and (', \\' in line or lines[i + 1].strip().startswith('patch')):
                # This is a multi-line with statement, add stream_chat_response before client.messages.create
                pass  # We need to reformat

# Actually, simpler approach - just add stream_chat_response patch right after client.messages.create
with open('tests/test_claude_light.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Pattern: patch("claude_light.llm.client.messages.create", return_value=
# Replace with: patch("claude_light.llm.stream_chat_response", return_value=streaming_resp), \
#               patch("claude_light.llm.client.messages.create", return_value=

pattern = r'(with patch\("claude_light\.llm\.stream_chat_response"[^,]*\), \\?\s*)?patch\("claude_light\.llm\.client\.messages\.create"'
matches = re.finditer(pattern, content)
matches_list = list(matches)

# If pattern already has stream_chat_response, don't add it again
# Otherwise add it

result = content
offset = 0

for match in matches_list:
    # Check if stream_chat_response is already in this with block
    with_block_start = content.rfind('with ', 0, match.start())
    colon_pos = content.find(':', match.start())
    with_block = content[with_block_start:colon_pos]
    
    if 'stream_chat_response' not in with_block:
        # Find the position to insert
        insert_pos = match.start() + offset
        # Check if we need to use _make_streaming_response
        # For now, let's create the streaming_response variable inline
        
        # Get the test function context
        test_start = content.rfind('def test_', 0, match.start())
        test_func = content[test_start:match.start()]
        
        if '_make_streaming_response' in test_func or 'streaming_response' in test_func:
            # We already have the streaming response created
            prefix = 'patch("claude_light.llm.stream_chat_response", return_value=streaming_response), \n             '
            result = result[:insert_pos] + prefix + result[insert_pos:]
            offset += len(prefix)

with open('tests/test_claude_light.py', 'w', encoding='utf-8') as f:
    f.write(result)

print(f"Updated {len(matches_list)} test patches")
