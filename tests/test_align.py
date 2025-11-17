import cv2
from source.align_images import align_images
import os

def test_align_samples_exist():
    img_path = os.path.join('samples', 'image.jpg')
    tpl_path = os.path.join('samples', 'template.jpg')
    assert os.path.exists(img_path), 'sample image missing'
    assert os.path.exists(tpl_path), 'sample template missing'

def test_align_runs_and_returns_image():
    img = cv2.imread('samples/image.jpg')
    tpl = cv2.imread('samples/template.jpg')
    aligned = align_images(img, tpl, debug=False)
    assert aligned is not None
    assert aligned.shape[0] == tpl.shape[0]
    assert aligned.shape[1] == tpl.shape[1]
