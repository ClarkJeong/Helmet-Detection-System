import RPi.GPIO as GPIO
import argparse
import sys
import time
import smtplib

from email.mime.text import MIMEText
import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

# LED가 연결된 GPIO 핀 번호
LED_PIN = 18  
LED_PIN_R = 17 

# 초음파 센서 핀 번호 설정
TRIG_PIN = 23
ECHO_PIN = 24

# GPIO 설정
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.setup(LED_PIN_R, GPIO.OUT)

# 초음파 센서 GPIO 모드 설정
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

def send_email(sender_email, sender_password, recipient_email, subject, message):
    # 이메일 내용을 MIMEText 객체로 생성
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipient_email

    try:
        # SMTP 서버에 연결
        server = smtplib.SMTP('smtp.naver.com', 587)
        server.starttls()
        # 로그인 인증
        server.login(sender_email, sender_password)
        # 이메일 전송
        server.sendmail(sender_email, recipient_email, msg.as_string())
        print("이메일이 성공적으로 전송되었습니다.")
    except Exception as e:
        print("이메일 전송 중 오류가 발생했습니다:", str(e))
    finally:
        # SMTP 서버 연결 종료
        server.quit()



def turn_on_led():
    GPIO.output(LED_PIN, GPIO.HIGH)

def turn_on_led_R():
    GPIO.output(LED_PIN_R, GPIO.HIGH)

def turn_off_led():
    GPIO.output(LED_PIN, GPIO.LOW)

def turn_off_led_R():
    GPIO.output(LED_PIN_R, GPIO.LOW)

# 추가-초음파 센서로 거리 측정하는 함수
def measure_distance():
    GPIO.output(TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, False)

    pulse_start = time.time()
    pulse_end = time.time()

    while GPIO.input(ECHO_PIN) == 0:
        pulse_start = time.time()

    while GPIO.input(ECHO_PIN) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    distance = round(distance, 2)  # 소수점 2자리까지 반올림
    return distance


# Visualization parameters
_ROW_SIZE = 20  # pixels
_LEFT_MARGIN = 24  # pixels
_TEXT_COLOR = (0, 0, 255)  # red
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_FPS_AVERAGE_FRAME_COUNT = 10


def run(model: str, max_results: int, score_threshold: float, num_threads: int,
        enable_edgetpu: bool, camera_id: int, width: int, height: int) -> None:
    """Continuously run inference on images acquired from the camera.

    Args:
        model: Name of the TFLite image classification model.
        max_results: Max of classification results.
        score_threshold: The score threshold of classification results.
        num_threads: Number of CPU threads to run the model.
        enable_edgetpu: Whether to run the model on EdgeTPU.
        camera_id: The camera id to be passed to OpenCV.
        width: The width of the frame captured from the camera.
        height: The height of the frame captured from the camera.
    """

    # Initialize the image classification model
    base_options = core.BaseOptions(
        file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)

    # Enable Coral by this setting
    classification_options = processor.ClassificationOptions(
        max_results=max_results, score_threshold=score_threshold)
    options = vision.ImageClassifierOptions(
        base_options=base_options, classification_options=classification_options)

    classifier = vision.ImageClassifier.create_from_options(options)

    # Variables to calculate FPS
    counter, fps = 0, 0
    start_time = time.time()

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)



    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        counter += 1
        image = cv2.flip(image, 1)

        # 초음파 센서로 거리 측정
        distance = measure_distance()
        print(f"Distance: {distance} cm")

        if distance <= 30:
            # Convert the image from BGR to RGB as required by the TFLite model.
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Create TensorImage from the RGB image
            tensor_image = vision.TensorImage.create_from_array(rgb_image)
            # List classification results
            categories = classifier.classify(tensor_image)

            # Check if 'crash helmet' is detected among the classifications
            crash_helmet_detected = False
            for category in categories.classifications[0].categories:
                if category.category_name == 'crash helmet':
                    crash_helmet_detected = True
                    break

            # Return 1 if 'crash helmet' is detected, otherwise 0
            result = 1 if crash_helmet_detected else 0

            turn_on_led_R()
            
            if result == 1:
                turn_on_led()
                turn_off_led_R()
            else:
                turn_off_led()
                turn_on_led_R()

            # Show the result on the image
            result_text = 'Crash helmet detected: {}'.format(crash_helmet_detected)
            text_location = (_LEFT_MARGIN, 2 * _ROW_SIZE)
            cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

            # Calculate the FPS
            if counter % _FPS_AVERAGE_FRAME_COUNT == 0:
                end_time = time.time()
                fps = _FPS_AVERAGE_FRAME_COUNT / (end_time - start_time)
                start_time = time.time()

            # Show the FPS
            fps_text = 'FPS = ' + str(int(fps))
            text_location = (_LEFT_MARGIN, _ROW_SIZE)
            cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

            # Show the image
            cv2.imshow('image_classification', image)
            
        # 대기 시간 추가 (1초)
        time.sleep(1)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1000) == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()

    turn_off_led()
    turn_off_led_R()

    # Send the email
    sender_email = " "
    sender_password = " "
    recipient_email = " "
    email_subject = "Notification: Program Stopped"
    email_message = "The program has been stopped."
    send_email(sender_email, sender_password, recipient_email, email_subject, email_message)
    

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Name of image classification model.',
        required=False,
        default='efficientnet_lite0.tflite')
    parser.add_argument(
        '--maxResults',
        help='Max of classification results.',
        required=False,
        default=3)
    parser.add_argument(
        '--scoreThreshold',
        help='The score threshold of classification results.',
        required=False,
        type=float,
        default=0.0)
    parser.add_argument(
        '--numThreads',
        help='Number of CPU threads to run the model.',
        required=False,
        default=4)
    parser.add_argument(
        '--enableEdgeTPU',
        help='Whether to run the model on EdgeTPU.',
        action='store_true',
        required=False,
        default=False)
    parser.add_argument(
        '--cameraId', help='Id of camera.', required=False, default=0)
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        default=640)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        default=480)
    args = parser.parse_args()

    run(args.model, int(args.maxResults),
        args.scoreThreshold, int(args.numThreads), bool(args.enableEdgeTPU),
        int(args.cameraId), args.frameWidth, args.frameHeight)


if __name__ == '__main__':
    main()
