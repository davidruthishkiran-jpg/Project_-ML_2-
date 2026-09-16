from flask import Flask, render_template
from load_data import get_data_summary
from placement_eda import run_eda
from ml_models import (
    run_linear_regression,
    run_logistic_regression,
    run_ridge_lasso,
    run_decision_tree,
    run_random_forest,
    run_xgboost,
    run_kmeans,
    run_model_comparison,
)

app = Flask(__name__)

_ML_RUNNERS = {
    "linear_regression":   run_linear_regression,
    "logistic_regression": run_logistic_regression,
    "ridge_lasso":         run_ridge_lasso,
    "decision_tree":       run_decision_tree,
    "random_forest":       run_random_forest,
    "xgboost":             run_xgboost,
    "kmeans":              run_kmeans,
    "model_comparison":    run_model_comparison,
}


@app.route("/")
def index():
    # Landing page, no section selected yet
    return render_template("index.html", active="none")


@app.route("/data-loading")
def data_loading():
    """Loads the dataset (server-side) and renders the summary into the page."""
    error = None
    summary = None
    try:
        summary = get_data_summary()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="data-loading",
        summary=summary,
        error=error,
    )
@app.route("/eda")
def eda():
    """Runs exploratory data analysis and renders results."""
    error = None
    eda_output = None
    try:
        eda_output = run_eda()   # call your EDA function
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "eda.html",
        active="eda",
        results=eda_output,
        error=error,
    )

@app.route("/ml")
def ml_home():
    """ML landing page — no algorithm selected yet."""
    return render_template("ml.html", active="ml", algo=None, result=None, error=None)


@app.route("/ml/<algo>")
def ml_run(algo):
    """Run a specific ML algorithm and render its charts."""
    runner = _ML_RUNNERS.get(algo)
    if runner is None:
        return render_template(
            "ml.html", active="ml", algo=algo, result=None,
            error=f"Unknown algorithm: '{algo}'"
        )
    error = None
    result = None
    try:
        result = runner()
    except Exception as exc:
        error = f"Error running {algo}: {exc}"
    return render_template("ml.html", active="ml", algo=algo, result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True)
