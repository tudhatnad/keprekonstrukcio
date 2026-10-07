import cv2

image = cv2.imread("horse-icons-26440.png", cv2.IMREAD_GRAYSCALE)

_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
binary_image = cv2.bitwise_not(binary_image)
cv2.imwrite("lo_kep_binarized.tif", binary_image)