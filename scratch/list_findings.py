import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

def parse_report(filepath, title):
    print(f"\n==========================================")
    print(f"REPORT: {title}")
    print(f"==========================================")
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Find markdown sections like ### or **[something]** or P0/P1/P2
    # Let's search for lines containing P0, P1, P2 or bullet points that look like findings.
    lines = content.split('\n')
    current_finding = []
    in_finding = False
    
    for idx, line in enumerate(lines):
        # A finding section typically starts with a header containing P0, P1, P2 or **[
        is_header = line.strip().startswith('###') or '**[' in line or '**P0' in line or '**P1' in line or '**P2' in line or line.strip().startswith('- **')
        
        if is_header:
            if current_finding:
                # Print previous finding if it contains P0 or P1
                full_text = '\n'.join(current_finding)
                if any(p in full_text for p in ['P0', 'P1', 'P2']):
                    print(full_text)
                    print("-" * 50)
                current_finding = []
            in_finding = True
            current_finding.append(line)
        elif in_finding:
            # Append lines until next header
            if line.strip() == "---" or line.strip().startswith('## '):
                in_finding = False
                full_text = '\n'.join(current_finding)
                if any(p in full_text for p in ['P0', 'P1', 'P2']):
                    print(full_text)
                    print("-" * 50)
                current_finding = []
            else:
                current_finding.append(line)
                
    if current_finding:
        full_text = '\n'.join(current_finding)
        if any(p in full_text for p in ['P0', 'P1', 'P2']):
            print(full_text)
            print("-" * 50)

parse_report("c:/Users/user/.gemini/antigravity/scratch/domain_2_3_untruncated.txt", "Domain 2+3 (Concurrency, Networking, AI, Swarm)")
parse_report("c:/Users/user/.gemini/antigravity/scratch/domain_4_5_untruncated.txt", "Domain 4+5 (Quant Math, ML, Risk, Execution)")
