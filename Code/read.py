import cv2 as cv

#read image
img = cv.imread('Photos/Yoshi Wallpaper.jpeg')
cv.imshow('Yoshi', img)

if cv.waitKey(0) == ord('q'):
      cv.destroyAllWindows()

#read video
capture = cv.VideoCapture('Photos/Dog.mp4')

fps = capture.get(cv.CAP_PROP_FPS)
delay = int(1000 / fps)

while True:
      isTrue, frame = capture.read()
      if not isTrue:
            break

      cv.imshow('Dog', frame)

      if cv.waitKey(delay) & 0xFF == ord('d'):
            break
            
capture.release()
cv.destroyAllWindows()