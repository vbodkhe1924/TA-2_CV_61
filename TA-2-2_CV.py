import cv2
import matplotlib.pyplot as plt
import numpy as np

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

img_all_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, 
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

mask_matches = mask.ravel().tolist()

ransac_matches = [good_matches[i] for i in range(len(good_matches)) if mask_matches[i]]

img_ransac_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, ransac_matches, None, 
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img_all_matches, cv2.COLOR_BGR2RGB))
plt.title(f'All Matches: {len(good_matches)} matches')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_ransac_matches, cv2.COLOR_BGR2RGB))
plt.title(f'After RANSAC: {len(ransac_matches)} inlier matches')
plt.axis('off')

plt.tight_layout()
plt.savefig('ransac_matches.png', dpi=300, bbox_inches='tight')
plt.show()

h, w = img1.shape[:2]
warped_img = cv2.warpPerspective(img1, H, (w, h))

plt.figure(figsize=(16, 8))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title('Image 1 (Source)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))
plt.title('Warped Image 1')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.title('Image 2 (Target)')
plt.axis('off')

plt.tight_layout()
plt.savefig('ransac_warp.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Total matches before RANSAC: {len(good_matches)}")
print(f"Inlier matches after RANSAC: {len(ransac_matches)}")
print(f"Homography matrix:\n{H}")

