import json
import os
import sys

import requests


# ============================================================
# GITHUB REPOSITORY CONFIGURATION
# ============================================================

GITHUB_OWNER = "PrathamModesara"

GITHUB_REPOSITORY = (
    "MLOPS-Image-Classification-Pipeline-with-Drift-Triggered-Retraining"
)

GITHUB_EVENT = "drift-alert"


# ============================================================
# SEND GITHUB DISPATCH
# ============================================================

def send_github_dispatch(drift_result):

    token = os.getenv("GITHUB_TOKEN")

    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN environment variable is not set."
        )

    url = (
        f"https://api.github.com/repos/"
        f"{GITHUB_OWNER}/"
        f"{GITHUB_REPOSITORY}/dispatches"
    )

    payload = {
        "event_type": GITHUB_EVENT,
        "client_payload": {
            "drift_score": drift_result["drift_score"],
            "threshold": drift_result["threshold"],
            "drift_detected": drift_result["drift_detected"],
            "dataset_type": drift_result["dataset_type"],
        },
    }

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2026-03-10",
        "Content-Type": "application/json",
    }

    print("=" * 60)
    print("GITHUB DRIFT ALERT")
    print("=" * 60)

    print(
        f"Repository : "
        f"{GITHUB_OWNER}/{GITHUB_REPOSITORY}"
    )

    print(
        f"Event      : {GITHUB_EVENT}"
    )

    print(
        f"Drift score: "
        f"{drift_result['drift_score']:.4f}"
    )

    print(
        f"Threshold  : "
        f"{drift_result['threshold']:.4f}"
    )

    print(
        f"Drift      : "
        f"{drift_result['drift_detected']}"
    )

    print(
        "\nSending repository_dispatch..."
    )

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=30,
    )

    if response.status_code != 204:

        print(
            "\nGitHub dispatch failed."
        )

        print(
            "Status:",
            response.status_code,
        )

        print(
            "Response:",
            response.text,
        )

        response.raise_for_status()

    print(
        "\nGitHub repository_dispatch sent successfully."
    )

    print(
        "Event:",
        GITHUB_EVENT,
    )

    print(
        "GitHub Actions should now start."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    drift_result_path = (
        "artifacts/drift_result.json"
    )

    if not os.path.exists(
        drift_result_path
    ):

        print(
            f"Drift result not found: "
            f"{drift_result_path}"
        )

        sys.exit(1)

    with open(
        drift_result_path,
        "r",
    ) as file:

        drift_result = json.load(file)

    print(
        "Loaded drift result:"
    )

    print(
        json.dumps(
            drift_result,
            indent=4,
        )
    )

    if not drift_result.get(
        "drift_detected",
        False,
    ):

        print(
            "\nNo drift detected."
        )

        print(
            "GitHub dispatch will NOT be sent."
        )

        return

    send_github_dispatch(
        drift_result
    )


if __name__ == "__main__":
    main()
