import pandas as pd
import matplotlib.pyplot as plt
import io
import cv2
import numpy as np
import re
from collections import deque
def morph_operation(img_gray, op='open', ksize=3, shape='rect', iterations=1):
  """
  Perform morphological opening/closing on a binary image derived from img_gray using Otsu.
  Params:
    - op: 'open' or 'close'
    - ksize: odd kernel size >= 1
    - shape: 'rect' | 'ellipse' | 'cross'
    - iterations: number of iterations (>=1)
  Returns: binary image after morphology (uint8 0/255)
  """
  img = img_gray.copy()
  if img.ndim == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # Binarize with Otsu
  _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

  # Normalize params
  k = max(1, int(ksize))
  if k % 2 == 0:
    k += 1
  iters = max(1, int(iterations))
  shape_map = {
    'rect': cv2.MORPH_RECT,
    'ellipse': cv2.MORPH_ELLIPSE,
    'cross': cv2.MORPH_CROSS
  }
  st_shape = shape_map.get(str(shape).lower(), cv2.MORPH_RECT)
  kernel = cv2.getStructuringElement(st_shape, (k, k))

  if str(op).lower() in ['open', 'opening', 'morph_open']:
    result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iters)
  else:
    result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iters)

  return result

# Bài 1a: Phân ngưỡng toàn cục
def global_thresholding(img_gray, epsilon=1):
  img = img_gray.copy()
  T = (np.min(img) + np.max(img)) // 2
  while True:
    G1 = img[img <= T]
    G2 = img[img > T]

    mu1 = np.mean(G1) if len(G1) > 0 else 0
    mu2 = np.mean(G2) if len(G2) > 0 else 0

    T_new = (mu1+mu2)/2
    if abs(T_new - T) < epsilon:
      break
    T = T_new

  return T


def Thresholding(img, T=100):
  img_temp = img.copy()
  img_temp[img_temp >= T] = 255
  img_temp[img_temp < T] = 0
  return img_temp

# Bài 1b: Phân ngưỡng thích nghi
def adaptiveThreshold(img, block_size=15, C=5):
    image = img.copy()
    h, w = image.shape
    binary = np.zeros((h, w), dtype=np.uint8)
    r = block_size // 2
    for y in range(h):
        for x in range(w):
            y1, y2 = max(0, y-r), min(h, y+r+1)
            x1, x2 = max(0, x-r), min(w, x+r+1)

            neighbor = image[y1:y2, x1:x2]
            T = np.mean(neighbor) - C

            if image[y, x] > T:
                binary[y, x] = 255
            else:
                binary[y, x] = 0
    return binary

# Bài 1c: Otsu's Method
def otsu(img_gray):
  # Bước 1: Tính xác suất xuất hiện mỗi mức xám
  img = img_gray.copy()
  hist, bins = np.histogram(img.flatten(), bins=256, range=(0, 256))
  hist = hist / len(img.flatten())
  np.sum(hist)


  # Bước 2: Tính trọng số 2 lớp, mean và mean toàn ảnh
  # a. Trọng số 2 lớp
  # w0 là sum từ 0->i, còn w1 là sum từ i+1 tới 255
  w0, w1 = np.zeros(256), np.zeros(256)
  for T in range(256):
    w0[T] = np.sum(hist[:T])
    w1[T] = np.sum(hist[T:])

  # b. Mean mỗi lớp
  phi0, phi1 = np.zeros(256), np.zeros(256)
  for T in range(256):
    if w0[T] == 0:
      phi0[T] = 0
    else:
      phi0[T] = np.sum(hist[:T] * np.arange(256)[:T])/w0[T]

    if w1[T] == 0:
      phi1[T] = 0
    else:
      phi1[T] = np.sum(hist[T:] * np.arange(256)[T:])/w1[T]

  # c. Trung bình toàn ảnh
  phi = np.sum(hist * np.arange(256))

  # Bước 3: Tính Between-Class Variance
  bcv = np.zeros(256)
  for T in range(256):
    bcv[T] = w0[T] * w1[T] * (phi0[T] - phi1[T])**2

  # Bước 4: Chọn T mà bcv lớn nhất:
  T = np.argmax(bcv)
  return T

# Bài 2: Region Growing
dX = [-1, 0, 1]
dY = [-1, 0, 1]

def regionGrowing(img_gray, seed, T=10):
  img = img_gray.copy()
  h, w = img.shape
  visited = np.zeros_like(img, dtype=bool)
  res = np.zeros_like(img, dtype=np.uint8)
  curX, curY = seed
  queue = deque([(curX, curY)])
  visited[curY, curX] = True

  while queue:
    x, y = queue.popleft()
    res[y, x] = 255

    for sX in dX:
      for sY in dY:
        if(sX == 0 and sY == 0):
          continue
        nX, nY = x + sX, y + sY
        if 0 <= nX < w and 0 <= nY < h:
          if visited[nY, nX]:
            continue
          if(abs(int(img[nY, nX]) - int(img[curY, curX])) < T):
            queue.append((nX, nY))
          visited[nY, nX] = True

  return res


# Bài 3: Watershed
def watershed_segment(img_gray, blur_ksize=3, dist_ratio = 0.1):
  ksize = (blur_ksize, blur_ksize)
  img_1 = img_gray.copy()
  # 1. Làm mượt ảnh để giảm nhiễu
  blur = cv2.GaussianBlur(img_1, ksize, 0)

  # 2. Threshold tự động (Otsu)
  _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
  # 3. Dùng morphology
  kernel = np.ones((3, 3), np.uint8)
  opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
  plt.imshow(opening, cmap="gray")

  # closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

  # 4. Sure background
  sure_bg = cv2.dilate(opening, kernel, iterations=3)

  # 5. Sure foreground
  dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
  _, sure_fg = cv2.threshold(dist, dist_ratio * dist.max(), 255, 0)
  sure_fg = sure_fg.astype(np.uint8)

  # 6. Unknown
  unknown = cv2.subtract(sure_bg, sure_fg)

  # 7. Đánh nhãn foreground
  _, markers = cv2.connectedComponents(sure_fg)
  markers = markers + 1
  markers[unknown == 255] = 0

  # 8. Watershed
  bgr = cv2.cvtColor(img_1, cv2.COLOR_GRAY2BGR)
  markers_ws = cv2.watershed(bgr, markers)

  # 9. Tô viền
  result_img = cv2.cvtColor(img_1, cv2.COLOR_GRAY2BGR)
  result_img[markers_ws == -1] = [255, 0, 0]  # biên = đỏ
  boundary_mask = (markers_ws == -1)

  return markers_ws, result_img, boundary_mask


# Bài 4a: Connected Components
def detect_by_connected_components(img_gray, output_prefix="connected"):
    # 1️⃣ Làm mượt ảnh và nhị phân hóa (Otsu)
    blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2️⃣ Dùng Connected Components để gán nhãn từng vùng trắng liên thông
    num_labels, labels = cv2.connectedComponents(thresh)

    # 3️⃣ Tô màu ngẫu nhiên cho từng nhãn để hiển thị
    label_img = np.zeros((*labels.shape, 3), dtype=np.uint8)
    colors = [tuple(np.random.randint(0,255,3).tolist()) for _ in range(num_labels)]
    for i in range(1, num_labels):  # Bỏ label 0 (nền)
        label_img[labels == i] = colors[i]

    # 4️⃣ Hiển thị kết quả
    print(f"[ConnectedComponents] Số đối tượng phát hiện: {num_labels - 1}")
    return num_labels - 1, label_img

# Bài 4b: Contour Detection
def detect_by_contours(img_gray, output_prefix="contour"):
    # 1️⃣ Tiền xử lý – nhị phân hóa ảnh
    blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2️⃣ Tìm các đường bao (contour)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3️⃣ Vẽ contour, bounding box, và convex hull
    contour_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    for i, cnt in enumerate(contours):
        # Vẽ contour
        cv2.drawContours(contour_img, [cnt], -1, (0,255,0), 2)

        # Bounding box
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(contour_img, (x,y), (x+w,y+h), (255,0,0), 1)

        # Convex hull (đường bao lồi)
        hull = cv2.convexHull(cnt)
        cv2.drawContours(contour_img, [hull], -1, (0,0,255), 1)

    # 4️⃣ Xuất tọa độ contour ra file (giữ nguyên hành vi cũ)
    try:
        with open(f"{output_prefix}_coords.txt", "w") as f:
            for i, cnt in enumerate(contours):
                f.write(f"Contour {i+1}:\n{cnt.reshape(-1,2)}\n\n")
    except Exception:
        # Không chặn nếu không ghi được file; sẽ trả về chuỗi để tải về qua web
        pass

    print(f"[Contours] Số đối tượng phát hiện: {len(contours)}")
    return len(contours), contour_img, contours

# Bài 5: Biểu diễn biên
directions = [
      (0, 1),   # 0: phải
      (-1, 1),  # 1: phải trên
      (-1, 0),  # 2: lên
      (-1, -1), # 3: trái trên
      (0, -1),  # 4: trái
      (1, -1),  # 5: trái dưới
      (1, 0),   # 6: xuống
      (1, 1)    # 7: phải dưới
  ]

def compute_chain_code(binary_img):
  img = binary_img.copy()
  _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
  img = thresh

  contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  cnt = max(contours, key=cv2.contourArea)


  res = []
  for i in range(1, len(cnt)):
      dx = cnt[i][0][1] - cnt[i-1][0][1]
      dy = cnt[i][0][0] - cnt[i-1][0][0]
      for code, (dy_dir, dx_dir) in enumerate(directions):
          if (dy, dx) == (dy_dir, dx_dir):
              res.append(code)
              break

  print(f"[ChainCode] Chiều dài mã hóa: {len(res)} bước")
  return res, cnt

def draw_chain_code(chain_code, start=(2, 2), figsize=(5, 5), color='red', show_grid=True):
  """
  Vẽ chain code bằng matplotlib và trả về ảnh (numpy array) để hiển thị trên web.

  Args:
    chain_code: Danh sách số (0..7) hoặc chuỗi "01234567...".
    start: Điểm bắt đầu theo định dạng (y, x) để giữ tương thích với code hiện tại.
    figsize: Kích thước figure matplotlib.
    color: Màu đường vẽ.
    show_grid: Bật/tắt lưới nền.

  Returns:
    np.ndarray (BGR) ảnh đã render từ matplotlib.
  """
  # Chuẩn hóa chuỗi chain code thành list[int]
  if isinstance(chain_code, (list, tuple, np.ndarray)):
    codes = [int(c) for c in chain_code if str(c).isdigit() and 0 <= int(c) <= 7]
  else:
    s = str(chain_code)
    codes = [int(ch) for ch in s if ch in '01234567']

  # Tính danh sách điểm (y, x)
  y, x = start
  points = [(y, x)]
  for code in codes:
    dy, dx = directions[code]
    y += dy
    x += dx
    points.append((y, x))

  ys, xs = zip(*points)

  # Vẽ bằng matplotlib nhưng không show, lưu vào buffer
  fig = plt.figure(figsize=figsize)
  ax = fig.add_subplot(111)
  ax.plot(xs, ys, '-o', color=color, markersize=2)
  ax.invert_yaxis()
  ax.set_aspect('equal', 'box')
  ax.grid(show_grid)
  # Ẩn trục cho gọn gàng
  ax.set_xticks([])
  ax.set_yticks([])

  buf = io.BytesIO()
  fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
  plt.close(fig)
  buf.seek(0)
  arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
  img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
  return img
  
def parse_contour_coords_text(text: str):
  """
  Parse contour coordinates from a text file content.
  Supported formats:
  - One point per line: "x y" or "x,y"
  - Blocks like:
      Contour 1:
      [[x y]
       [x y]
       ...]
    possibly multiple such blocks.

  Returns: List of numpy arrays with shape (n,1,2), dtype=int32 suitable for cv2.polylines/drawContours.
  """
  text = text.strip()
  if not text:
    return []

  contours = []

  # If it contains labeled blocks, split by 'Contour N:'
  if re.search(r"Contour\s+\d+\s*:", text, flags=re.IGNORECASE):
    pattern = re.compile(r"Contour\s+\d+\s*:\s*([\s\S]*?)(?=Contour\s+\d+\s*:|$)", re.IGNORECASE)
    for block in pattern.findall(text):
      pts = re.findall(r"(\d+)\s*[ ,]+\s*(\d+)", block)
      if pts:
        arr = np.array([[ [int(x), int(y)] ] for x, y in pts], dtype=np.int32)
        contours.append(arr)
  else:
    # Try simple list of pairs in whole text
    pts = re.findall(r"(\d+)\s*[ ,]+\s*(\d+)", text)
    if pts:
      arr = np.array([[ [int(x), int(y)] ] for x, y in pts], dtype=np.int32)
      contours.append(arr)

  return contours

def draw_contours_on_image(img_bgr: np.ndarray, contours, color=(0,255,0), thickness=2, closed=True):
  """
  Draw one or more polylines/contours onto the given BGR image copy and return it.
  contours: list of ndarray with shape (n,1,2) in int32.
  """
  if img_bgr is None:
    raise ValueError("draw_contours_on_image: img_bgr is None")
  out = img_bgr.copy()
  for cnt in contours or []:
    if isinstance(cnt, np.ndarray) and cnt.ndim >= 2 and cnt.shape[-1] == 2:
      cv2.polylines(out, [cnt.reshape(-1,1,2)], isClosed=bool(closed), color=color, thickness=thickness)
  return out

def contours_to_text(contours) -> str:
  """Format contours to a text similar to the saved file output."""
  if contours is None:
    return ""
  parts = []
  for i, cnt in enumerate(contours, start=1):
    try:
      arr = cnt.reshape(-1, 2)
      parts.append(f"Contour {i}:\n{arr}\n\n")
    except Exception:
      continue
  return ''.join(parts)
  
  
# Helper
def compute_iou(pred_mask, gt_mask):
  # Đưa về mask nhị phân 0/1
  pred = (pred_mask > 0).astype(np.uint8)
  gt = (gt_mask > 0).astype(np.uint8)

  intersection = np.logical_and(pred, gt).sum()
  union = np.logical_or(pred, gt).sum()

  if union == 0:
      return 1.0 if intersection == 0 else 0.0

  return intersection / union