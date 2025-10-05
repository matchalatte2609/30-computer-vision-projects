import cv2
from PIL import Image
from util import get_limits

yellow = [0, 255, 255] #BGR value of harro the pig
webcam = cv2.VideoCapture(0)
while True:
    ret, frame = webcam.read()

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_limit, upper_limit = get_limits(color=yellow)

    mask_yellow = cv2.inRange(hsvImage, lower_limit, upper_limit)

    # Apply morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)

    mask_ = Image.fromarray(mask_yellow)
    bbox = mask_.getbbox() # boundary box

    if bbox is not None:
        x1, y1, x2, y2 = bbox

        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
