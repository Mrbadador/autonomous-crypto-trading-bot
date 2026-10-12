import os
import re

def clean_content(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    # Decode double escapes if any (e.g. \n, \")
    try:
        # If it is wrapped in JSON quotes, unwrap it
        if content.startswith('"') and content.endswith('"'):
            content = json.loads(content)
        else:
            # Replace literal escapes
            content = content.encode('utf-8').decode('unicode_escape')
    except Exception as e:
        # Fallback to simple replace
        content = content.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
    return content

content_2_3 = clean_content("c:/Users/user/.gemini/antigravity/scratch/domain_2_3_untruncated.txt")
content_4_5 = clean_content("c:/Users/user/.gemini/antigravity/scratch/domain_4_5_untruncated.txt")

with open("c:/Users/user/.gemini/antigravity/scratch/domain_2_3_clean.md", "w", encoding="utf-8") as f:
    f.write(content_2_3)

with open("c:/Users/user/.gemini/antigravity/scratch/domain_4_5_clean.md", "w", encoding="utf-8") as f:
    f.write(content_4_5)

print("Saved clean markdown files to scratch folder.")
