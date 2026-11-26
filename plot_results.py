import re
import matplotlib.pyplot as plt

def parse_results(filename="results.txt"):
    data = {}
    current_graph = None
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("GRAPH:"):
                current_graph = line.split("GRAPH:")[1].strip()
                data[current_graph] = {"OpenMP": ([], []), "OpenCilk": ([], []), "Pthreads": ([], [])}
            elif line.startswith("|") and current_graph:
                parts = [p.strip() for p in line.split("|")[1:-1]]
                if len(parts) == 10 and parts[0] in data[current_graph]:
                    model = parts[0]
                    threads = int(parts[1])
                    speedup = float(parts[9].replace("x", ""))
                    data[current_graph][model][0].append(threads)
                    data[current_graph][model][1].append(speedup)
    return data

def main():
    data = parse_results("results.txt")
    graphs = list(data.keys())

    fig, axes = plt.subplots(1, len(graphs), figsize=(13, 3.8), sharey=True)
    if len(graphs) == 1:
        axes = [axes]

    styles = {
        "OpenMP":   {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
        "OpenCilk": {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
        "Pthreads": {"color": "#2ca02c", "marker": "^", "linestyle": "-."},
    }

    for ax, graph in zip(axes, graphs):
        for model, (threads, speedups) in data[graph].items():
            ax.plot(threads, speedups, label=model, linewidth=2, markersize=6, **styles[model])

        # Κάθετη διακεκομμένη γραμμή στα 8 physical cores & 16 hardware threads
        ax.axvline(8, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        ax.axvline(16, color="red", linestyle=":", linewidth=1, alpha=0.6)

        ax.set_title(graph, fontsize=11, fontweight="bold")
        ax.set_xlabel("Threads", fontsize=10)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Speedup (vs. Sequential)", fontsize=10)
    axes[-1].legend(loc="lower right", frameon=True, fontsize=9)

    fig.suptitle("Parallel Speedup Scaling Across Benchmarks (Intel Core 7 240H: 8C/16T)", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("speedup_plots.pdf", bbox_inches="tight")
    plt.savefig("speedup_plots.png", dpi=300, bbox_inches="tight")
    print("Saved speedup_plots.pdf and speedup_plots.png")

if __name__ == "__main__":
    main()
