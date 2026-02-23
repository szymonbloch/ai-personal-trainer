import cv2
from typing import List


def find_available_cameras(max_tested: int = 5) -> List[int]:
    """
    Scans for available camera indices by attempting to open them with OpenCV.

    Args:
        max_tested (int): The maximum number of indices to test (from 0 to max_tested - 1).

    Returns:
        List[int]: A list of available camera indices.
    """
    available_cameras = []

    for idx in range(max_tested):
        # cv2.CAP_DSHOW is used to prevent slow camera initialization on Windows
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)

        if cap.isOpened():
            print(f"✅ Camera available at index {idx}")
            available_cameras.append(idx)
            cap.release()
        else:
            print(f"❌ No camera found at index {idx}")

    return available_cameras


def main() -> None:
    """Main execution function for camera detection."""
    print("Scanning for available cameras...\n")
    available = find_available_cameras()

    if available:
        print(f"\nSummary: Found {len(available)} camera(s) at indices: {available}")
    else:
        print("\nSummary: No cameras detected on this system.")


if __name__ == "__main__":
    main()