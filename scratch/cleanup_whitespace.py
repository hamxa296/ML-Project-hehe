import re

with open('reports/project_report.tex', 'r') as f:
    content = f.read()

# Replace 3 or more newlines with 2 newlines (one blank line)
content = re.sub(r'\n{3,}', '\n\n', content)

with open('reports/project_report.tex', 'w') as f:
    f.write(content)
