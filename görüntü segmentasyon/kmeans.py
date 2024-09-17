import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('HarryPotter.jpg')

# Görüntüyü yeniden boyutlandır 
img = cv2.resize(img, (600, 400))

# K-means Kümeleme için, görüntüyü (3 kanal) vektöre çevir
Z = img.reshape((-1, 3))
Z = np.float32(Z)

# K-means kriterleri (maksimum iterasyon sayısı ve epsilon)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Küme sayısı (K) 
K = 3

# K-means algoritması
_, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Sonuçları yeniden biçimlendir 
center = np.uint8(center)
res = center[label.flatten()]
segmented_image = res.reshape((img.shape))


plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Orijinal Görüntü')
plt.subplot(122), plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)), plt.title('Segmentasyon Sonucu')
plt.show()
