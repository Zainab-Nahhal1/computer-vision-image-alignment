import cv2
import numpy as np
import imutils

def align_images(image, template, maxFeatures=500, keepPercent=0.2, debug=False):
    """Align `image` to `template` using ORB feature matching and homography.

    Args:
        image: BGR image (np.ndarray) - source image to be aligned.
        template: BGR image (np.ndarray) - target template image.
        maxFeatures: int - max ORB features to detect.
        keepPercent: float - fraction of matches to keep (0-1).
        debug: bool - if True, shows matched keypoints.

    Returns:
        aligned: warped image aligned to template size.
    """
    if image is None:
        raise ValueError("Input image is None. Check your path.")
    if template is None:
        raise ValueError("Template image is None. Check your path.")

    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(maxFeatures)
    kpsA, descA = orb.detectAndCompute(imageGray, None)
    kpsB, descB = orb.detectAndCompute(templateGray, None)

    if descA is None or descB is None:
        raise RuntimeError("No descriptors found in one of the images.")

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descA, descB)
    matches = sorted(matches, key=lambda x: x.distance)

    keep = max(4, int(len(matches) * keepPercent))  # at least 4 matches
    matches = matches[:keep]

    pointsA = np.zeros((len(matches), 2), dtype=np.float32)
    pointsB = np.zeros((len(matches), 2), dtype=np.float32)

    for i, m in enumerate(matches):
        pointsA[i] = kpsA[m.queryIdx].pt
        pointsB[i] = kpsB[m.trainIdx].pt

    if debug:
        matchedVis = cv2.drawMatches(image, kpsA, template, kpsB, matches, None)
        matchedVis = imutils.resize(matchedVis, width=1000)
        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.waitKey(0)

    H, status = cv2.findHomography(pointsA, pointsB, cv2.RANSAC)
    if H is None:
        raise RuntimeError("Homography could not be computed.")

    h, w = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))
    return aligned
