import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread("B1.jpeg")
img2 = cv2.imread("B2.jpeg")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, 
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(16, 8))
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.title(f'SIFT Matching: {len(good_matches)} good matches found')
plt.savefig('sift_matches.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Total keypoints in image 1: {len(keypoints1)}")
print(f"Total keypoints in image 2: {len(keypoints2)}")
print(f"Number of good matches: {len(good_matches)}")
