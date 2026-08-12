import json
import os
import subprocess

from src.config import (
    DRIFT_RESULT,
    DRIFT_THRESHOLD,
)


def check_drift():

    print("=" * 60)
    print("DRIFT TRIGGER CHECK")
    print("=" * 60)

    # --------------------------------------------------------
    # Check drift result
    # --------------------------------------------------------

    if not os.path.exists(DRIFT_RESULT):

        print(
            "\nDrift result not found."
        )

        print(
            "Run the drift detector first."
        )

        return False

    with open(
        DRIFT_RESULT,
        "r",
    ) as file:

        result = json.load(file)

    drift_score = float(
        result["drift_score"]
    )

    threshold = float(
        result["threshold"]
    )

    drift_detected = bool(
        result["drift_detected"]
    )

    print(
        f"\nDrift Score : "
        f"{drift_score:.4f}"
    )

    print(
        f"Threshold   : "
        f"{threshold:.4f}"
    )

    print(
        f"Drift       : "
        f"{drift_detected}"
    )

    # --------------------------------------------------------
    # Trigger retraining
    # --------------------------------------------------------

    if drift_detected:

        print(
            "\n" + "!" * 60
        )

        print(
            "DRIFT DETECTED"
        )

        print(
            "Starting model retraining..."
        )

        print(
            "!" * 60
        )

        # ----------------------------------------------------
        # Run ZenML pipeline
        # ----------------------------------------------------

        result = subprocess.run(
            [
                "python",
                "-m",
                "src.zenml_pipeline",
            ],
            check=False,
        )

        if result.returncode == 0:

            print(
                "\n" + "=" * 60
            )

            print(
                "RETRAINING COMPLETED"
            )

            print(
                "=" * 60
            )

            return True

        else:

            print(
                "\n" + "=" * 60
            )

            print(
                "RETRAINING FAILED"
            )

            print(
                "=" * 60
            )

            return False

    # --------------------------------------------------------
    # No drift
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "NO DRIFT DETECTED"
    )

    print(
        "Retraining is not required."
    )

    print(
        "=" * 60
    )

    return False


if __name__ == "__main__":

    check_drift()
