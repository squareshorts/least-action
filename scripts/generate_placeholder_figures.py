import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

def generate_benchmark_map():
    # Example values from summary
    models = ["Minimum Jerk", "Condition Mean", "Spline Condition", "Condition Action", "Semantic Action", "Trial-Fitted Action"]
    raw_nll = [44.29, 15.38, 33.61, 38.81, 38.17, 31.39]
    semantic_gen = [0, 0, 0, 0, 0.69, 0] # Example gain
    latent_validity = [0, 0, 0, 1, 1, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['gray' if l == 0 else 'blue' for l in latent_validity]

    scatter = ax.scatter(raw_nll, semantic_gen, c=colors, s=100)
    for i, txt in enumerate(models):
        ax.annotate(txt, (raw_nll[i], semantic_gen[i]), xytext=(5, 5), textcoords='offset points')

    ax.set_xlabel("Raw Held-Out NLL (Lower is better, but watch for overfitting)")
    ax.set_ylabel("Semantic Generalization (LOIO NLL Gain)")
    ax.grid(True, linestyle='--', alpha=0.6)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Has Latent Validity (\u03C1)', markerfacecolor='blue', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='No Latent Validity', markerfacecolor='gray', markersize=10)]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.savefig(FIGURES / "benchmark_value_map.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_independent_margin_placeholder():
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.text(0.5, 0.5, "Independent norming data not included\nin this submission",
            horizontalalignment='center', verticalalignment='center', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=1'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Independent Semantic Margin")
    ax.set_ylabel("Fitted Competitor Attraction ($\\rho$)")
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.savefig(FIGURES / "independent_margin_rho.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    generate_benchmark_map()
    generate_independent_margin_placeholder()
    print("Generated placeholder figures in figures/")
