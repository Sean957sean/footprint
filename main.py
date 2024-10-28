import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess_image(file_path):
    # 이미지 로드 및 그레이스케일 변환
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    # 이진화하여 발자국 영역을 강조
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    return img, binary

def find_contours(binary_image):
    # 윤곽선 찾기
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def find_extreme_points(contour):
    # 기존 각도 계산에 사용할 극점 (y좌표가 가장 높은 점, 가장 낮은 점, x좌표가 가장 작은 점)
    topmost = tuple(contour[contour[:, :, 1].argmin()][0])  # y좌표가 가장 높은 점
    bottommost = tuple(contour[contour[:, :, 1].argmax()][0])  # y좌표가 가장 낮은 점
    leftmost = tuple(contour[contour[:, :, 0].argmin()][0])  # x좌표가 가장 작은 점
    # 새 기준에 사용할 극점 (y좌표가 가장 높은 점, 가장 낮은 점, x좌표가 가장 큰 점)
    rightmost = tuple(contour[contour[:, :, 0].argmax()][0])  # x좌표가 가장 큰 점
    return topmost, bottommost, leftmost, rightmost

def calculate_angle(p1, p2, p3):
    # 세 점을 통해 각도를 계산
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ab = a - b
    cb = c - b
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def draw_and_calculate_average_angle(img, contours):
    # 윤곽선 그리기
    contour_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)

    # 첫 번째 윤곽선에서 극점 찾기
    topmost, bottommost, leftmost, rightmost = find_extreme_points(contours[0])

    # 기존 기준 각도 계산 (y 좌표 최고, 최저, x 좌표 최소)
    angle1 = calculate_angle(topmost, bottommost, leftmost)
    # 새로운 기준 각도 계산 (y 좌표 최고, 최저, x 좌표 최대)
    angle2 = calculate_angle(topmost, bottommost, rightmost)

    # 각도의 평균 계산
    average_angle = (angle1 + angle2) / 2

    # 각도 및 평균 각도 텍스트 표시
    cv2.line(contour_img, topmost, bottommost, (255, 0, 0), 2)  # 파란색 선
    cv2.line(contour_img, bottommost, leftmost, (0, 255, 255), 2)  # 노란색 선
    cv2.line(contour_img, bottommost, rightmost, (255, 0, 255), 2)  # 분홍색 선

    cv2.putText(contour_img, f"Angle1: {angle1:.1f}", (bottommost[0] + 10, bottommost[1] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(contour_img, f"Angle2: {angle2:.1f}", (bottommost[0] + 10, bottommost[1] + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    cv2.putText(contour_img, f"Average Angle: {average_angle:.1f}", (bottommost[0] + 10, bottommost[1] + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return contour_img, average_angle

# 메인 코드
file_path = "fossil2.png"
original_img, binary_img = load_and_preprocess_image(file_path)
contours = find_contours(binary_img)
img_with_average_angle, average_angle = draw_and_calculate_average_angle(original_img, contours)

# 결과 시각화
plt.imshow(img_with_average_angle)
plt.title("Contours with Key Points and Average Angle")
plt.axis('off')
plt.show()

# 평균 각도 출력

print("Average Angle between extreme points:", average_angle)
if float(average_angle) < 40: 
    print("theropod")
else:
    print("ornithischian")
