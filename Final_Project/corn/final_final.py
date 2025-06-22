import cv2 as cv
from ultralytics import YOLO
import RPi.GPIO as GPIO
from time import sleep, time
import threading

# ========== [ GPIO 핀 설정 ] ==========
# --- 스텝 모터 핀 ---
STEP_PUL_1 = 20  # 첫 번째 모터 Pulse 핀
STEP_DIR_1 = 21  # 첫 번째 모터 Direction 핀

STEP_PUL_2 = 27  # 두 번째 모터 Pulse 핀
STEP_DIR_2 = 22  # 두 번째 모터 Direction 핀

# --- 서보 모터 핀 ---
SERVO_PIN = 18

# --- 서보 각도 설정용 duty 값 범위 ---
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3

# ========== [ GPIO 초기 설정 ] ==========
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)

# 스텝 모터 1
GPIO.setup(STEP_PUL_1, GPIO.OUT)
GPIO.setup(STEP_DIR_1, GPIO.OUT)
GPIO.output(STEP_DIR_1, GPIO.LOW)

# 스텝 모터 2
GPIO.setup(STEP_PUL_2, GPIO.OUT)
GPIO.setup(STEP_DIR_2, GPIO.OUT)
GPIO.output(STEP_DIR_2, GPIO.HIGH)

# 서보 모터 PWM 설정 (50Hz)
pwm = GPIO.PWM(SERVO_PIN, 50)
pwm.start(0)

# ========== [ 서보 제어 함수 ] ==========
def setServoPos(degree):
    """각도를 PWM duty로 변환하여 서보 회전"""
    if degree > 180.:
        degree = 180.
    elif degree < 0:
        degree = 0
    duty = SERVO_MIN_DUTY + (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    print("Degree: {} → Duty: {}".format(degree, duty))
    pwm.ChangeDutyCycle(duty)
    sleep(0.5)
    pwm.ChangeDutyCycle(0)

# ========== [ 스텝 모터 백그라운드 동작 쓰레드 (각각) ] ==========
def step_motor_loop_1():
    try:
        while True:
            GPIO.output(STEP_PUL_1, GPIO.HIGH)
            sleep(0.08)
            GPIO.output(STEP_PUL_1, GPIO.LOW)
            sleep(0.08)
    except:
        pass

def step_motor_loop_2():
    try:
        while True:
            GPIO.output(STEP_PUL_2, GPIO.HIGH)
            sleep(0.20)  # 두 번째 모터는 더 빠르게 움직이는 예
            GPIO.output(STEP_PUL_2, GPIO.LOW)
            sleep(0.20)
    except:
        pass

# 스텝 모터 쓰레드 시작
step_thread_1 = threading.Thread(target=step_motor_loop_1, daemon=True)
step_thread_2 = threading.Thread(target=step_motor_loop_2, daemon=True)
step_thread_1.start()
step_thread_2.start()

# ========== [ YOLO 모델 불러오기 ] ==========
model = YOLO('/home/hwang/corn/runs/detect/train/weights/best.pt')

# 카메라 초기화
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print('카메라를 열 수 없습니다.')
    GPIO.cleanup()
    exit()

# ========== [ 변수 초기화 ] ==========
TOP_THRESHOLD = 230
BOTTOM_THRESHOLD = TOP_THRESHOLD - 15
prev_time = time()
prev_angle = 90
setServoPos(prev_angle)

# ========== [ 메인 루프 ] ==========
try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        curr_time = time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        results = model(frame)
        annotated_frame = results[0].plot()

        # 기준선 시각화
        cv.line(annotated_frame, (0, TOP_THRESHOLD), (annotated_frame.shape[1], TOP_THRESHOLD), (255, 0, 0), 2)
        cv.line(annotated_frame, (0, BOTTOM_THRESHOLD), (annotated_frame.shape[1], BOTTOM_THRESHOLD), (255, 0, 0), 2)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0]

                if BOTTOM_THRESHOLD < y1 < TOP_THRESHOLD:
                    if cls_id == 0:
                        target_angle = 10
                    elif cls_id == 1:
                        target_angle = 65
                    elif cls_id == 2:
                        target_angle = 115
                    elif cls_id == 3:
                        target_angle = 170
                    else:
                        continue

                    if target_angle != prev_angle:
                        setServoPos(target_angle)
                        prev_angle = target_angle

        cv.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.putText(annotated_frame, f"Angle: {prev_angle}", (10, 70),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv.imshow("YOLOv8 + Servo + 2 Step Motors", annotated_frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\n사용자 종료 요청 (Ctrl+C)")

finally:
    cap.release()
    cv.destroyAllWindows()
    pwm.stop()
    GPIO.cleanup()
    print("모든 장치 정리 완료.")

