#!/usr/bin/env python3
"""Run a demo alignment using images in samples/"""
import cv2
import imutils
from source.align_images import align_images

def main():
    image = cv2.imread("samples/image.jpg")
    template = cv2.imread("samples/template.jpg")

    aligned = align_images(image, template, debug=True)
    aligned = imutils.resize(aligned, width=700)
    template = imutils.resize(template, width=700)
    stacked = cv2.hconcat([aligned, template])
    cv2.imshow("Aligned Image", stacked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
