import nbformat
from nbformat import NotebookNode

def convert_ipynb_to_py(ipynb_path: str, py_path: str, md_prefix: str = "# "):
    # Load the notebook
    nb = nbformat.read(ipynb_path, as_version=4)
    
    lines = []
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            # Prefix each markdown line so it becomes a comment
            for line in cell.source.splitlines():
                lines.append(f"{md_prefix}{line}")
            lines.append("")  # blank line between cells
        elif cell.cell_type == 'code':
            # Include code as-is
            lines.append(cell.source)
            lines.append("")  # blank line between cells
    
    # Write out the .py file
    with open(py_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python convert.py notebook.ipynb script.py")
    else:
        convert_ipynb_to_py(sys.argv[1], sys.argv[2])
        print(f"Converted {sys.argv[1]} → {sys.argv[2]}")
