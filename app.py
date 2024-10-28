from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# 이미지 분석 함수
def analyze_image(file_path):
    # 이미지 로드 및 전처리
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    # 윤곽선 검출
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 윤곽선에서 극점 찾기
    topmost = tuple(contours[0][contours[0][:, :, 1].argmin()][0])
    bottommost = tuple(contours[0][contours[0][:, :, 1].argmax()][0])
    leftmost = tuple(contours[0][contours[0][:, :, 0].argmin()][0])
    rightmost = tuple(contours[0][contours[0][:, :, 0].argmax()][0])

    # 각도 계산 함수
    def calculate_angle(p1, p2, p3):
        a, b, c = np.array(p1), np.array(p2), np.array(p3)
        ab, cb = a - b, c - b
        cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
        angle = np.degrees(np.arccos(cosine_angle))
        return angle
    
    # 기존 기준 각도와 새로운 기준 각도 계산
    angle1 = calculate_angle(topmost, bottommost, leftmost)
    angle2 = calculate_angle(topmost, bottommost, rightmost)
    average_angle = (angle1 + angle2) / 2

    # 결과 이미지에 윤곽선 및 각도 표시
    contour_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)
    cv2.line(contour_img, topmost, bottommost, (255, 0, 0), 2)
    cv2.line(contour_img, bottommost, leftmost, (0, 255, 255), 2)
    cv2.line(contour_img, bottommost, rightmost, (255, 0, 255), 2)
    cv2.putText(contour_img, f"Avg Angle: {average_angle:.1f}", (bottommost[0] + 10, bottommost[1] + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # 처리된 이미지 저장
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.png')
    cv2.imwrite(result_path, contour_img)
    
    return result_path, average_angle

# 라우트 설정
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            result_path, average_angle = analyze_image(filepath)
            return render_template('index.html', result_img=result_path, angle=average_angle)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
