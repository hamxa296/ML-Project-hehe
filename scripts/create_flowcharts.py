#!/usr/bin/env python3
"""Generate large, legible flowchart PNGs for the report.

Outputs:
  project/artifacts/report_figures/flowcharts/
    - system_architecture.png
    - methodology_flow.png
    - prefect_orchestration.png
    - containerization_workflow.png
"""
from pathlib import Path

try:
    from graphviz import Digraph
except Exception:
    Digraph = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
except Exception:
    plt = None


OUT_DIR = Path("project/artifacts/report_figures/flowcharts")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def render_dot(dot: "Digraph", name: str):
    out_path = OUT_DIR / f"{name}.png"
    try:
        png = dot.pipe(format="png")
        out_path.write_bytes(png)
        print(f"Wrote {out_path}")
    except Exception as e:
        print(f"Failed to render {name}: {e}")


def render_matplotlib(boxes, edges, name, size=(11,6), dpi=200):
    out_path = OUT_DIR / f"{name}.png"
    figw, figh = size
    if plt is None:
        print(f"matplotlib not available; cannot render {name}")
        return
    fig, ax = plt.subplots(figsize=(figw, figh), dpi=dpi)
    ax.axis('off')
    for key, (x, y, w, h, label, color) in boxes.items():
        box = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle='round,pad=0.3',
                             linewidth=1.5, edgecolor=color, facecolor=color + '20')
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=10, wrap=True)
    for a, b in edges:
        xa, ya = boxes[a][0], boxes[a][1]
        xb, yb = boxes[b][0], boxes[b][1]
        arrow = FancyArrowPatch((xa + 0.2, ya), (xb - 0.2, yb), arrowstyle='->', mutation_scale=20, linewidth=1.2, color='#0066CC')
        ax.add_patch(arrow)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {out_path} (matplotlib)")


def make_system_architecture():
    if Digraph is not None:
        d = Digraph("system", format="png")
        d.attr(rankdir="LR", dpi="300", size="11,6")
        d.attr('node', shape='rect', style='filled,rounded', fillcolor='#E6F0FF', color='#0066CC', fontsize='12')
        d.node('CI', 'CI/CD\n(GitHub Actions)')
        d.node('Prefect', 'Prefect\nOrchestration')
        d.node('Training', 'Training Pipeline\n(XGBoost + KMeans)')
        d.node('Registry', 'Model Registry\n(Artifacts)')
        d.node('API', 'FastAPI\nInference')
        d.node('Dashboard', 'React Dashboard')
        d.edge('CI', 'Prefect')
        d.edge('Prefect', 'Training')
        d.edge('Training', 'Registry')
        d.edge('Registry', 'API')
        d.edge('API', 'Dashboard')
        render_dot(d, 'system_architecture')
    else:
        boxes = {
            'CI': (1.5, 3.0, 2.8, 0.9, 'CI/CD\n(GitHub Actions)', '#0066CC'),
            'Prefect': (4.0, 3.0, 2.8, 0.9, 'Prefect\nOrchestration', '#0066CC'),
            'Training': (6.5, 3.0, 2.8, 0.9, 'Training\n(XGBoost + KMeans)', '#0066CC'),
            'Registry': (8.5, 3.0, 2.8, 0.9, 'Model Registry\n(Artifacts)', '#0066CC'),
            'API': (8.5, 1.7, 2.6, 0.9, 'FastAPI\nInference', '#0066CC'),
            'Dashboard': (10.5, 1.7, 2.6, 0.9, 'React Dashboard', '#0066CC'),
        }
        edges = [('CI', 'Prefect'), ('Prefect', 'Training'), ('Training', 'Registry'), ('Registry', 'API'), ('API', 'Dashboard')]
        render_matplotlib(boxes, edges, 'system_architecture', size=(11,6), dpi=200)


def make_methodology_flow():
    if Digraph is not None:
        d = Digraph('methodology', format='png')
        d.attr(rankdir='LR', dpi='300', size='11,6')
        d.attr('node', shape='rect', style='filled,rounded', fillcolor='#F0F8FF', color='#0066CC', fontsize='11')
        d.node('Raw', 'Raw CSVs\n(train_transaction.csv + train_identity.csv)')
        d.node('Merge', 'Merge & Memory Downcast')
        d.node('Feat', 'Feature Engineering\n(time, freq enc, log)')
        d.node('Prune', 'Pruning (>70% nulls)')
        d.node('Cluster', 'KMeans\n(n=5)')
        d.node('Train', 'Train XGBoost\n(scale_pos_weight)')
        d.node('Eval', 'Evaluate\n(PR/ROC, Confusion)')
        d.node('Serve', 'Save & Serve\n(model_latest.pkl)')
        edges = [('Raw','Merge'),('Merge','Feat'),('Feat','Prune'),('Prune','Cluster'),('Cluster','Train'),('Train','Eval'),('Eval','Serve')]
        for a,b in edges:
            d.edge(a,b)
        render_dot(d, 'methodology_flow')
    else:
        boxes = {
            'Raw': (1.0, 3.0, 2.6, 0.9, 'Raw CSVs\n(train_transaction.csv + train_identity.csv)', '#0066CC'),
            'Merge': (3.0, 3.0, 2.2, 0.9, 'Merge & Memory Downcast', '#0066CC'),
            'Feat': (5.0, 3.0, 2.6, 0.9, 'Feature Engineering\n(time, freq enc, log)', '#0066CC'),
            'Prune': (7.0, 3.0, 2.4, 0.9, 'Pruning (>70% nulls)', '#0066CC'),
            'Cluster': (9.0, 3.0, 2.2, 0.9, 'KMeans\n(n=5)', '#0066CC'),
            'Train': (5.0, 1.6, 2.6, 0.9, 'Train XGBoost\n(scale_pos_weight)', '#0066CC'),
            'Eval': (7.5, 1.6, 2.4, 0.9, 'Evaluate\n(PR/ROC, Confusion)', '#0066CC'),
            'Serve': (9.5, 1.6, 2.6, 0.9, 'Save & Serve\n(model_latest.pkl)', '#0066CC'),
        }
        edges = [('Raw','Merge'),('Merge','Feat'),('Feat','Prune'),('Prune','Cluster'),('Cluster','Train'),('Train','Eval'),('Eval','Serve')]
        render_matplotlib(boxes, edges, 'methodology_flow', size=(11,6), dpi=200)


def make_prefect_orchestration():
    if Digraph is not None:
        d = Digraph('prefect', format='png')
        d.attr(rankdir='TB', dpi='300', size='8,6')
        d.attr('node', shape='rect', style='filled,rounded', fillcolor='#FFF7E6', color='#CC6600', fontsize='11')
        d.node('ingest', 'Ingest')
        d.node('eda', 'Raw EDA')
        d.node('preproc', 'Preprocessing')
        d.node('train', 'Training')
        d.node('eval', 'Evaluation')
        d.node('publish', 'Publish Artifacts')
        d.edges([('ingest','eda'),('eda','preproc'),('preproc','train'),('train','eval'),('eval','publish')])
        render_dot(d, 'prefect_orchestration')
    else:
        boxes = {
            'ingest': (5.5, 5.0, 2.2, 0.8, 'Ingest', '#CC6600'),
            'eda': (5.5, 4.0, 2.2, 0.8, 'Raw EDA', '#CC6600'),
            'preproc': (5.5, 3.0, 2.6, 0.8, 'Preprocessing', '#CC6600'),
            'train': (5.5, 2.0, 2.2, 0.8, 'Training', '#CC6600'),
            'eval': (5.5, 1.0, 2.2, 0.8, 'Evaluation', '#CC6600'),
            'publish': (5.5, 0.0, 2.6, 0.8, 'Publish Artifacts', '#CC6600'),
        }
        edges = [('ingest','eda'),('eda','preproc'),('preproc','train'),('train','eval'),('eval','publish')]
        render_matplotlib(boxes, edges, 'prefect_orchestration', size=(11,6), dpi=200)


def make_containerization_workflow():
    if Digraph is not None:
        d = Digraph('container', format='png')
        d.attr(rankdir='LR', dpi='300', size='11,4')
        d.attr('node', shape='rect', style='filled,rounded', fillcolor='#F7FFF0', color='#2E8B57', fontsize='11')
        d.node('dev', 'Developer')
        d.node('ci', 'CI Runner\n(GitHub Actions)')
        d.node('build', 'Docker Build')
        d.node('push', 'Registry\n(Container Image)')
        d.node('deploy', 'Production')
        d.edges([('dev','ci'),('ci','build'),('build','push'),('push','deploy')])
        render_dot(d, 'containerization_workflow')
    else:
        boxes = {
            'dev': (1.5, 2.0, 2.4, 0.8, 'Developer', '#2E8B57'),
            'ci': (4.0, 2.0, 2.6, 0.8, 'CI Runner\n(GitHub Actions)', '#2E8B57'),
            'build': (6.5, 2.0, 2.6, 0.8, 'Docker Build', '#2E8B57'),
            'push': (8.5, 2.0, 2.6, 0.8, 'Registry\n(Container Image)', '#2E8B57'),
            'deploy': (10.0, 2.0, 2.4, 0.8, 'Production', '#2E8B57'),
        }
        edges = [('dev','ci'),('ci','build'),('build','push'),('push','deploy')]
        render_matplotlib(boxes, edges, 'containerization_workflow', size=(11,4), dpi=200)


def main():
    # Try graphviz first; if unavailable, fallback to matplotlib rendering
    if Digraph is None:
        print('graphviz python library not available; falling back to matplotlib rendering if available.')
    make_system_architecture()
    make_methodology_flow()
    make_prefect_orchestration()
    make_containerization_workflow()


if __name__ == '__main__':
    main()
