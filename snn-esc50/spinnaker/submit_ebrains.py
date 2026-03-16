"""
Submit SpiNNaker inference job to EBRAINS Neuromorphic Computing Platform.

This script uses the nmpi Python client to submit the SNN inference
job to SpiNNaker via EBRAINS.

Prerequisites:
    1. EBRAINS account (register at https://ebrains.eu/register)
    2. Request neuromorphic access by emailing
       neuromorphic@humanbrainproject.eu with your EBRAINS username
    3. Install the client: pip install hbp_neuromorphic_platform
    4. Have weight files ready in results/spinnaker_weights/

Usage:
    python submit_ebrains.py --username YOUR_EBRAINS_USERNAME --collab-id YOUR_COLLAB_ID

    Or with saved token:
    python submit_ebrains.py --username YOUR_EBRAINS_USERNAME --token YOUR_TOKEN --collab-id YOUR_COLLAB_ID
"""

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path


def check_prerequisites():
    """Check that all required files and packages are available."""
    weights_dir = Path(__file__).parent.parent / "results" / "spinnaker_weights"

    required_files = [
        "fc1_connections.npy",
        "fc2_connections.npy",
        "test_spike_features.npy",
        "test_labels.npy",
        "metadata.json",
    ]

    missing = []
    for f in required_files:
        if not (weights_dir / f).exists():
            missing.append(f)

    if missing:
        print("Missing required weight/data files:")
        for f in missing:
            print(f"  - {weights_dir / f}")
        print("\nRun these scripts first:")
        print("  python spinnaker/convert_weights.py --model-path results/snn/direct/best_fold4.pt")
        print("  python spinnaker/extract_features.py --model-path results/snn/direct/best_fold4.pt")
        return False

    try:
        import nmpi
        print(f"nmpi client version available")
    except ImportError:
        print("nmpi client not installed.")
        print("Install with: pip install hbp_neuromorphic_platform")
        return False

    return True


def prepare_job_directory():
    """Prepare a temporary directory with all files needed for the job."""
    weights_dir = Path(__file__).parent.parent / "results" / "spinnaker_weights"
    job_script = Path(__file__).parent / "ebrains_job.py"

    # Create temp directory
    job_dir = tempfile.mkdtemp(prefix="spinnaker_esc50_")

    # Copy job script
    shutil.copy2(job_script, os.path.join(job_dir, "run.py"))

    # Copy weight and data files
    for f in ["fc1_connections.npy", "fc2_connections.npy",
              "test_spike_features.npy", "test_labels.npy", "metadata.json"]:
        shutil.copy2(weights_dir / f, os.path.join(job_dir, f))

    print(f"Job directory prepared: {job_dir}")
    for f in sorted(os.listdir(job_dir)):
        size = os.path.getsize(os.path.join(job_dir, f))
        print(f"  {f} ({size / 1024:.1f} KB)")

    return job_dir


def submit_job(username, token=None, collab_id=None, wait=True):
    """Submit the inference job to SpiNNaker via EBRAINS.

    Args:
        username: EBRAINS username.
        token: Authentication token (optional, will prompt for password).
        collab_id: EBRAINS Collab ID.
        wait: If True, block until job completes.
    """
    import nmpi

    # Authenticate
    if token:
        client = nmpi.Client(username, token=token)
    else:
        print(f"Authenticating as {username}...")
        print("(You will be prompted for your EBRAINS password)")
        client = nmpi.Client(username)

    print(f"Authenticated successfully. Token: {client.token[:20]}...")

    # Prepare job files
    job_dir = prepare_job_directory()

    # Submit job
    print(f"\nSubmitting job to SpiNNaker...")
    job_id = client.submit_job(
        source=job_dir,
        platform=nmpi.SPINNAKER,
        collab_id=collab_id,
        command="run.py {system}",
        tags=["esc50", "snn", "sound-classification"],
        wait=wait,
    )

    if wait:
        # Job completed (wait=True returns full job dict)
        print(f"\nJob completed!")
        print(f"Job ID: {job_id.get('id', 'unknown')}")
        print(f"Status: {job_id.get('status', 'unknown')}")

        # Download results
        results_dir = Path(__file__).parent.parent / "results" / "spinnaker_results"
        results_dir.mkdir(parents=True, exist_ok=True)

        try:
            client.download_data(job_id, local_dir=str(results_dir))
            print(f"Results downloaded to {results_dir}")
        except Exception as e:
            print(f"Could not download results: {e}")
            print("You can download manually from the EBRAINS Collaboratory")
    else:
        print(f"\nJob submitted! Job ID: {job_id}")
        print(f"Check status with: client.job_status('{job_id}')")
        print(f"Or view in EBRAINS Collaboratory Job Manager")

    # Clean up temp directory
    shutil.rmtree(job_dir, ignore_errors=True)

    return job_id


def main():
    parser = argparse.ArgumentParser(
        description="Submit SNN inference job to EBRAINS SpiNNaker"
    )
    parser.add_argument("--username", required=True,
                        help="EBRAINS username")
    parser.add_argument("--token", default=None,
                        help="EBRAINS authentication token (optional)")
    parser.add_argument("--collab-id", type=int, default=None,
                        help="EBRAINS Collab ID")
    parser.add_argument("--no-wait", action="store_true",
                        help="Submit and return immediately (don't wait)")
    parser.add_argument("--check-only", action="store_true",
                        help="Only check prerequisites, don't submit")
    args = parser.parse_args()

    print("=" * 60)
    print("EBRAINS SpiNNaker Job Submission")
    print("ESC-50 SNN Sound Classification")
    print("=" * 60)

    print("\nChecking prerequisites...")
    if not check_prerequisites():
        return

    if args.check_only:
        print("\nAll prerequisites met. Ready to submit.")
        return

    print()
    submit_job(
        username=args.username,
        token=args.token,
        collab_id=args.collab_id,
        wait=not args.no_wait,
    )


if __name__ == "__main__":
    main()
