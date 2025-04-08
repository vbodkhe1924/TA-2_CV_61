import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("B1.jpeg")
original_img = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

block_size = 2
aperture_size = 3
k = 0.04
threshold_ratio = 0.01

dst = cv2.cornerHarris(gray, block_size, aperture_size, k)

dst_norm = np.empty(dst.shape, dtype=np.float32)
cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

threshold = threshold_ratio * dst_norm.max()

corner_img = original_img.copy()

for i in range(dst_norm.shape[0]):
    for j in range(dst_norm.shape[1]):
        if dst_norm[i, j] > threshold:
            cv2.circle(corner_img, (j, i), 3, (0, 0, 255), 1)

corner_count = np.sum(dst_norm > threshold)

dst_norm_dilated = cv2.dilate(dst_norm, None)

heatmap = cv2.applyColorMap(np.uint8(dst_norm_dilated), cv2.COLORMAP_JET)

plt.figure(figsize=(16, 8))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(corner_img, cv2.COLOR_BGR2RGB))
plt.title(f'Harris Corners: {corner_count} corners detected')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
plt.title('Corner Response Heatmap')
plt.axis('off')

plt.tight_layout()
plt.savefig('harris_corners.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Number of corners detected: {corner_count}")
print(f"Parameters used: block_size={block_size}, aperture_size={aperture_size}, k={k}")