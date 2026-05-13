import re

with open('docs/relatorio/ADI 25_26.md', 'r', encoding='utf-8') as f:
    md_text = f.read()

lines = md_text.split('\n')
out = []
in_table = False
in_list = False
first_row = True

def escape_tex(text):
    text = text.replace("\\~", "~").replace("\\=", "=").replace("\\<", "<").replace("\\>", ">")
    text = text.replace("\\)", ")").replace("\\(", "(")
    text = text.replace("\\_", "_")
    text = text.replace("%", "\\%")
    text = text.replace("_", "\\_")
    text = text.replace("<", "$<$").replace(">", "$>$").replace("~", "$\\sim$")
    text = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', text)
    return text

for i, line in enumerate(lines):
    sline = line.strip()
    
    # 1) Metadata links (ignore completely)
    if sline.startswith("[image"):
        continue
        
    # Check end of list logic first so we don't skip headers
    if in_list:
        if sline == "":
            pass
        elif not sline.startswith("- ") and not re.match(r'^\d+\.', sline):
             for j in range(len(out)-1, -1, -1):
                 if "\\begin{itemize}" in out[j]:
                     out.append("\\end{itemize}\n")
                     break
                 if "\\begin{enumerate}" in out[j]:
                     out.append("\\end{enumerate}\n")
                     break
             in_list = False

    # Standalone headers formatted with **
    if sline.startswith("**") and sline.endswith("**") and len(sline) > 4:
        if not in_table and not in_list:
            res = "section"
            clean = sline.replace("**", "")
            if clean not in ["Definição da metodologia", "Conclusão", "Modelação e Avaliação"]:
                res = "subsection"
            out.append(f"\n\\{res}{{{escape_tex(clean)}}}\n")
            continue

    # Images
    if line.startswith("![][image"):
        img_id = re.search(r"image(\d+)", line).group(1)
        name_map = {
            '1': 'Visão geral do workflow desenvolvido no KNIME.png',
            '2': 'Exploração.png',
            '3': 'Transformação.png',
            '4': 'Decision Tree.png',
            '5': 'Random Forest.png',
            '6': 'Gradient Boosted Trees.png',
            '7': 'Multilayer Perceptron.png',
            '8': 'Keras.png'
        }
        fig_name = name_map.get(img_id, 'image.png')
        out.append("\\begin{figure}[H]")
        out.append("    \\centering")
        out.append(f"    \\includegraphics[width=0.9\\textwidth]{{{fig_name}}}")
        out.append(f"    \\caption{{{escape_tex(fig_name.replace('.png',''))}}}")
        out.append(f"    \\label{{fig:img{img_id}}}")
        out.append("\\end{figure}\n")
        continue

    # Lists
    if sline.startswith("- ") and not "|-" in line:
        if not in_list:
            out.append("\\begin{itemize}")
            in_list = True
        item = escape_tex(sline[2:])
        out.append(f"    \\item {item}")
        continue
    elif sline.startswith("1. ") or sline.startswith("2. ") or sline.startswith("3. "):
        if not in_list:
            out.append("\\begin{enumerate}")
            in_list = True
        item = escape_tex(sline[3:])
        out.append(f"    \\item {item}")
        continue

    # Tables
    if "|" in line:
        if "-|-" in line or "|---" in line or "---|" in line or ":---:" in line:
            continue
        if not in_table:
            in_table = True
            first_row = True
            max_cols = 0
            for j in range(i, len(lines)):
                if not "|" in lines[j]: break
                if "-|-" in lines[j] or "|---" in lines[j] or "---|" in lines[j] or ":---:" in lines[j]: continue
                cols = len(lines[j].split("|")) - 2
                if cols > max_cols: max_cols = cols

            align = "c" * max_cols
            out.append("\\begin{table}[H]")
            out.append("    \\centering")
            out.append(f"    \\begin{{tabular}}{{{align}}}")
            out.append("    \\hline")
            
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        clean_cells = [escape_tex(c) for c in cells]
        
        out.append("    " + " & ".join(clean_cells) + " \\\\")
        if first_row:
            out.append("    \\hline")
            first_row = False
        continue
    else:
        if in_table:
            out.append("    \\hline")
            out.append("    \\end{tabular}")
            out.append("\\end{table}\n")
            in_table = False

    # Regular text
    if not in_table and not in_list and not sline.startswith("**") and not line.startswith("![][image"):
         out.append(escape_tex(line))

if in_table:
    out.append("    \\hline")
    out.append("    \\end{tabular}")
    out.append("\\end{table}\n")

if in_list:
    for j in range(len(out)-1, -1, -1):
        if "\\begin{itemize}" in out[j]:
            out.append("\\end{itemize}\n")
            break
        if "\\begin{enumerate}" in out[j]:
            out.append("\\end{enumerate}\n")
            break

result = "\n".join(out)
result = result.replace("\\end{itemize}\n\n\\begin{itemize}", "")
result = result.replace("\\end{enumerate}\n\n\\begin{enumerate}", "")

with open('docs/relatorio/relatorio.tex', 'r', encoding='utf-8') as f:
    text_content = f.read()

prefix = r"""\part*{Parte II --- Tarefa 2: Dataset Atribuído (Consumo Energético)}
\addcontentsline{toc}{section}{Parte II --- Tarefa 2: Dataset Atribuído (Consumo Energético)}

"""
idx = text_content.find(r"\part*{Parte II")

new_content = text_content[:idx] + prefix + result + "\n\\end{document}\n"

with open('docs/relatorio/relatorio.tex', 'w', encoding='utf-8') as f:
    f.write(new_content)
