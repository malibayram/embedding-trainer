
path = '/root/.pyenv/versions/3.12.12/lib/python3.12/site-packages/transformers/tokenization_utils_base.py'
try:
    with open(path, 'r') as f:
        lines = f.readlines()

    # Check if we are at the right place
    target_line_idx = 1209 # 0-indexed, so this corresponds to line 1210
    
    # Verify the line content to be safe
    if "self.SPECIAL_TOKENS_ATTRIBUTES = self.SPECIAL_TOKENS_ATTRIBUTES + list(special_tokens.keys())" in lines[target_line_idx]:
        print("Found target line.")
        insert_content = "        if isinstance(special_tokens, list):\n            return\n"
        lines.insert(target_line_idx, insert_content)
        
        with open(path, 'w') as f:
            f.writelines(lines)
        print("Patched successfully.")
    else:
        print("Target line mismatch. Aborting.")
        print(f"Line at {target_line_idx+1}: {lines[target_line_idx]}")

except Exception as e:
    print(f"Error: {e}")
