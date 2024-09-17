import cv2
import numpy as np
class GrabCut:
    def __init__(self):
        image_path = 'image.jpg'
        self.image = cv2.imread(image_path)
        self.mask = np.zeros(self.image.shape[:2], np.uint8)
        self.background_model = np.zeros((1, 65), np.float64)
        self.front_model = np.zeros((1, 65), np.float64)

    def filter(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        sobel_combine = cv2.addWeighted(cv2.convertScaleAbs(sobel_x), 1, cv2.convertScaleAbs(sobel_y),1, 0)
        _, sobel_combined = cv2.threshold(sobel_combine, 10, 30, cv2.THRESH_BINARY)

        self.image = cv2.bitwise_and(self.image, self.image, mask=sobel_combined)

    def masking(self):
        self.mask2 = np.where((self.mask == 2) | (self.mask == 0), 0, 1).astype('uint8')
        self.result = self.image * self.mask2[:, :, np.newaxis]

    def rectangle(self, rect):
        self.rect = rect

    def segmentation(self):
        cv2.grabCut(self.image, self.mask, self.rect, self.background_model, self.front_model, 7, cv2.GC_INIT_WITH_RECT)
        self.masking()
    def result_image(self):
        cv2.imshow('Result-image', self.result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        output_path = 'output_result.jpg'  
        cv2.imwrite(output_path, self.result)
        print(f"Çıktı {output_path} dosyasına kaydedildi.")
def main():
    grabcut = GrabCut()
    grabcut.filter()
    rect = (18, 40, 600, 600)
    grabcut.rectangle(rect)
    grabcut.segmentation()
    grabcut.result_image()

main()