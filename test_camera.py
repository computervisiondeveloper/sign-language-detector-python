import cv2

def test_cameras():
    for i in range(5):  # Test indices 0 to 4
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera {i} is available.")
            cap.release()
        else:
            print(f"Camera {i} is not available.")

test_cameras()
