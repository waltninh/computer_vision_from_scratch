import cv2
import matplotlib.pyplot as plt
 
img = cv2.imread("person.jpg")                     # ảnh màu tải từ pexels.com
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)       # chuyển sang grayscale
 
canny = cv2.Canny(gray, 100, 200)                  # Canny edge detection
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobel_x, sobel_y)            # Sobel (gộp 2 hướng)
laplacian = cv2.Laplacian(gray, cv2.CV_64F)        # Laplacian
 
titles = ["Original (gray)", "Canny", "Sobel", "Laplacian"]
images = [gray, canny, sobel, abs(laplacian)]
for i, (t, im) in enumerate(zip(titles, images)):
    plt.subplot(1, 4, i + 1); plt.imshow(im, cmap="gray"); plt.title(t); plt.axis("off")
plt.show()
