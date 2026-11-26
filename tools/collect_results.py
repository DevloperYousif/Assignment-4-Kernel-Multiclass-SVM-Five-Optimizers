"""Collect the executed-notebook results into results/summary.json (used by the presentation and docs builders,
so every number on the slides comes from the result files written by the notebooks)."""
import json
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")


def rd(name, **kw):
    return pd.read_csv(os.path.join(R, name), **kw)


def main():
    comp = rd("optimizer_comparison.csv")
    out = {
        "hyper": json.load(open(os.path.join(R, "hyperparameters.json"))),
        "comparison": comp.to_dict(orient="records"),
        "verdict": json.load(open(os.path.join(R, "optimizer_verdict.json"))) if os.path.exists(os.path.join(R, "optimizer_verdict.json")) else {},
        "kernel_selection": rd("kernel_selection.csv").to_dict(orient="records"),
        "basis_size": rd("basis_size_analysis.csv").to_dict(orient="records"),
        "kernel_comparison": rd("kernel_comparison.csv").to_dict(orient="records"),
        "linear_vs_kernel": rd("linear_vs_kernel.csv", index_col=0).to_dict(),
        "per_class_f1": rd("per_class_f1.csv", index_col=0).to_dict() if os.path.exists(os.path.join(R, "per_class_f1.csv")) else {},
        "approximation": rd("approximation_quality.csv").to_dict(orient="records"),
        "exact_demo": rd("exact_kernel_demo.csv").to_dict(orient="records"),
        "resources": rd("resource_report.csv").to_dict(orient="records") if os.path.exists(os.path.join(R, "resource_report.csv")) else [],
        "batch_study": rd("batch_size_study.csv").to_dict(orient="records") if os.path.exists(os.path.join(R, "batch_size_study.csv")) else [],
        "lr_search": {f.split("_")[-1][:-4]: rd(f).to_dict(orient="records") for f in os.listdir(R) if f.startswith("lr_search_")},
    }
    from PIL import Image
    fig_dir = os.path.join(R, "figures")
    out["fig_sizes"] = {f: Image.open(os.path.join(fig_dir, f)).size for f in sorted(os.listdir(fig_dir)) if f.endswith(".png")}
    text = json.dumps(out, indent=1, default=float).replace(": NaN", ": null")
    open(os.path.join(R, "summary.json"), "w").write(text)
    print("wrote results/summary.json")


if __name__ == "__main__":
    main()
