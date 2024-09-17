import sys
import cv2
import numpy

# Komşu piksel değerlerini inceleyen fonksiyon
def neighbourhood(image, x, y):
    # Komşu piksellerin değerlerini bir sözlükte sakla
    neighbour_region_numbers = {}
    for i in range(-1, 2):  # x etrafındaki komşu pikselleri incele
        for j in range(-1, 2):  # y etrafındaki komşu pikselleri incele
            if (i == 0 and j == 0):
                continue  # Kendisiyle eşleşmeyi atla
            if (x+i < 0 or y+j < 0):  # Eğer koordinatlar görüntü sınırlarının dışında ise devam et
                continue
            if (x+i >= image.shape[0] or y+j >= image.shape[1]):  # Eğer koordinatlar görüntü sınırlarının dışında ise devam et
                continue
            # Eğer komşu bölge numarası sözlükte yoksa ekle, varsa sayısını artır
            if (neighbour_region_numbers.get(image[x+i][y+j]) == None):
                neighbour_region_numbers[image[x+i][y+j]] = 1  # Sözlükte yeni bir giriş oluştur
            else:
                neighbour_region_numbers[image[x+i][y+j]] += 1  # Sözlükte mevcut bir girişin sayısını artır

    # Eğer 0 anahtarı varsa (arka planı temsil eder), sil
    if (neighbour_region_numbers.get(0) != None):
        del neighbour_region_numbers[0]

    # Sözlüğün anahtarlarını (bölge numaralarını) al
    keys = list(neighbour_region_numbers)

    # Kontrolü kolaylaştırmak için anahtarları sırala
    keys.sort()

    if (keys[0] == -1):  # Eğer komşulardan biri ayrı bir bölgedeyse
        if (len(keys) == 1):  # Ayrı bir bölge
            return -1
        elif (len(keys) == 2):  # Başka bir bölgenin parçası
            return keys[1]
        else:  # Watershed (bölge sınırı)
            return 0
    else:
        if (len(keys) == 1):  # Başka bir bölgenin parçası
            return keys[0]
        else:  # Watershed (bölge sınırı)
            return 0

# Watershed algoritması ile segmentasyon yapan fonksiyon
def watershed_segmentation(image):
    # Piksel yoğunluklarını ve koordinatlarını içeren bir liste oluştur
    intensity_list = []
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            # Yoğunluk ve koordinatları listeye ekle
            intensity_list.append((image[x][y], (x, y)))

    # Listeyi piksel yoğunluklarına göre artan sırada sırala
    intensity_list.sort()

    # Segmentasyon yapılacak boş bir görüntü dizisi oluştur (-1 ile doldurulmuş)
    segmented_image = numpy.full(image.shape, -1, dtype=int)

    # Yoğunluk listesi üzerinde gezinerek segmentasyon işlemini gerçekleştir
    region_number = 0  # Bölge numarasını takip etmek için sayaç
    for i in range(len(intensity_list)):
        # Terminalde ilerlemeyi göstermek için iterasyon sayısını yazdır
        sys.stdout.write("\rPixel {} of {}...".format(i, len(intensity_list)))
        sys.stdout.flush()

        # Piksel yoğunluğunu ve koordinatlarını al
        intensity = intensity_list[i][0]
        x = intensity_list[i][1][0]
        y = intensity_list[i][1][1]

        # Mevcut pikselin komşu piksellerine göre bölge durumunu al
        region_status = neighbourhood(segmented_image, x, y)

        # Bölge numarasını veya watershed durumunu uygun şekilde ata
        if (region_status == -1):  # Ayrı bir bölge
            region_number += 1
            segmented_image[x][y] = region_number
        elif (region_status == 0):  # Watershed
            segmented_image[x][y] = 0
        else:  # Başka bir bölgenin parçası
            segmented_image[x][y] = region_status

    # Segmentasyon işlemi tamamlanan görüntüyü döndür
    return segmented_image


# Ana fonksiyon, programın başlangıç noktası
def main(argv):
    # Giriş görüntüsünü oku

    img = cv2.imread("image/1.png", 0)  # Gri tonlamalı olarak oku

    # Eğer görüntü okunamazsa hata ver
    if (img is None):
        print ("{} görüntüsü açılamadı")
        print ("Doğru görüntü yolunu sağladığınızdan emin olun")
        sys.exit(2)

    # Watershed segmentasyonu kullanarak giriş görüntüsünü bölgelere ayır
    segmented_image = watershed_segmentation(img)

    # Segmentasyon yapılan görüntüyü kaydet
    cv2.imwrite("image/target.png", segmented_image)

    # Orijinal ve segmentasyon yapılan görüntüyü yan yana göster
    input_image = cv2.resize(img, (0,0), None, 0.5, 0.5)  # Orijinal görüntüyü yeniden boyutlandır
    seg_image = cv2.resize(cv2.imread("image/target.png", 0), (0,0), None, 0.5, 0.5)  # Segmentasyon yapılan görüntüyü yeniden boyutlandır
    numpy_horiz = numpy.hstack((input_image, seg_image))  # İki görüntüyü yan yana ekle
    cv2.imshow('Orijinal görüntü ------------------------ Segmentasyon yapılmış görüntü', numpy_horiz)
    cv2.waitKey(0)  # Tuşa basılana kadar görüntüyü göster

# Program ana fonksiyondan çalıştırıldığında bu kod bloğu çalışacak
if __name__ == "__main__":
    main(sys.argv[1:])  # Komut satırından verilen argümanları alarak çalıştır
