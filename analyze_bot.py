import ast
import os
import sys

def check_file(filepath):
    issues = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # check if body only contains 'pass', '...', or raises NotImplementedError
                if not node.body:
                    continue
                first_stmt = node.body[0]
                is_empty = False
                
                # Check for just 'pass'
                if isinstance(first_stmt, ast.Pass) and len(node.body) == 1:
                    is_empty = True
                
                # Check for just docstring + pass
                elif len(node.body) == 2 and isinstance(first_stmt, ast.Expr) and isinstance(node.body[1], ast.Pass):
                    is_empty = True
                
                # Check for just '...' (Ellipsis)
                elif isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant) and first_stmt.value.value is Ellipsis:
                    is_empty = True

                # Check for NotImplementedError
                elif isinstance(first_stmt, ast.Raise):
                    if isinstance(first_stmt.exc, ast.Call) and isinstance(first_stmt.exc.func, ast.Name) and first_stmt.exc.func.id == 'NotImplementedError':
                        is_empty = True
                    elif isinstance(first_stmt.exc, ast.Name) and first_stmt.exc.id == 'NotImplementedError':
                        is_empty = True

                if is_empty:
                    issues.append(f"Line {node.lineno}: {node.name} is mocked/empty (contains pass or NotImplementedError)")
                    
    except SyntaxError as e:
        issues.append(f"Syntax Error: {e.msg} at line {e.lineno}")
    except Exception as e:
        pass
    return issues

def main():
    skip_dirs = ['.venv', '__pycache__', '.git', 'archive', 'data', 'logs']
    all_issues = {}
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for f in files:
            if f.endswith('.py'):
                path = os.path.join(root, f)
                res = check_file(path)
                if res:
                    all_issues[path] = res
                    
    for p, issues in all_issues.items():
        print(f"--- {p} ---")
        for i in issues:
            print(f"  {i}")

if __name__ == '__main__':
    main()
