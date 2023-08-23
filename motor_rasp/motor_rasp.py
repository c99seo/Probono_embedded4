import RPi.GPIO as GPIO
from time import sleep

#모터 상태
STOP = 0
FORWARD = 1
BACKWARD = 2
# motorpercent = 80

#모터 채널
CH1 = 0
CH2 = 1
CH3 = 2
CH4 = 3

#PIN 입출력 설정
OUTPUT = 1
INPUT = 0

#PIN 설정
HIGH = 1
LOW = 0

#실제 핀 정의
#PWM PIN
ENA = 26    #37 pin
ENB = 0     #27 pin
ENC = 17    #11 pin
END = 11    #23 pin

#GPIO PIN
motorA1 = 19    #37 pin
motorA2 = 13    #35 pin

motorB1 = 6     #31 pin
motorB2 = 5     #29 pin

motorC1 = 27    #13 pin
motorC2 = 22    #15 pin

motorD1 = 9     #21 pin
motorD2 = 10    #19 pin

#핀 설정 함수
def setPinConfig(EN, INA, INB):        
    GPIO.setup(EN, GPIO.OUT)
    GPIO.setup(INA, GPIO.OUT)
    GPIO.setup(INB, GPIO.OUT)
    # GPIO.setup(INC, GPIO.OUT)
    # GPIO.setup(IND, GPIO.OUT)
    # 100khz 로 PWM 동작 시킴 
    pwm = GPIO.PWM(EN, 100) 
    # 우선 PWM 멈춤.   
    pwm.start(0) 
    return pwm

#모터 제어 함수
def setMotorContorl(pwm, INA, INB, speed, stat):
    
    #모터 속도 제어 PWM
    pwm.ChangeDutyCycle(speed)
    
    if stat == FORWARD:
        GPIO.output(INA, HIGH)
        GPIO.output(INB, LOW)

    #뒤로
    elif stat == BACKWARD:
        GPIO.output(INA, LOW)
        GPIO.output(INB, HIGH)
        
    #정지
    elif stat == STOP:
        GPIO.output(INA, LOW)
        GPIO.output(INB, LOW)
        
        
        
# 모터 제어함수 간단하게 사용하기 위해 한번더 래핑(감쌈)
def setMotor(ch, speed, stat):
    if ch == CH1:
        #pwmA는 핀 설정 후 pwm 핸들을 리턴 받은 값이다.
        setMotorContorl(pwmA, motorA1, motorA2, speed, stat)
    elif ch == CH2:
        #pwmB는 핀 설정 후 pwm 핸들을 리턴 받은 값이다.
        setMotorContorl(pwmB, motorB1, motorB2, speed, stat)
    elif ch == CH3:
        setMotorContorl(pwmC, motorC1, motorC2, speed, stat)
    elif ch == CH4:
        setMotorContorl(pwmD, motorD1, motorD2, speed, stat)

# GPIO 모드 설정 
GPIO.setmode(GPIO.BCM)

#모터 핀 설정
#핀 설정후 PWM 핸들 얻어옴 
pwmA = setPinConfig(ENA, motorA1, motorA2)
pwmB = setPinConfig(ENB, motorB1, motorB2)
pwmC = setPinConfig(ENC, motorC1, motorC2)
pwmD = setPinConfig(END, motorD1, motorD2)

################### 참고 ###################
# # 모터 방향 설정
# def forward():
#     setMotor(CH1, 100, FORWARD)
#     setMotor(CH2, 100, FORWARD)
#     setMotor(CH3, 100, FORWARD)
#     setMotor(CH4, 100, FORWARD)
#     sleep(1)
#     setMotor(CH1, 0, FORWARD)
#     setMotor(CH2, 0, FORWARD)
#     setMotor(CH3, 0, FORWARD)
#     setMotor(CH4, 0, FORWARD)
    
# def back():
#     setMotor(CH1, 100, BACKWARD)
#     setMotor(CH2, 100, BACKWARD)
#     setMotor(CH3, 100, BACKWARD)
#     setMotor(CH4, 100, BACKWARD)
#     sleep(1)
#     setMotor(CH1, 0, BACKWARD)
#     setMotor(CH2, 0, BACKWARD)
#     setMotor(CH3, 0, BACKWARD)
#     setMotor(CH4, 0, BACKWARD)
    
# def stopp():
#     setMotor(CH1, 100, STOP)
#     setMotor(CH2, 100, STOP)
#     setMotor(CH3, 100, STOP)
#     setMotor(CH4, 100, STOP)
#     sleep(1)
#     setMotor(CH1, 100, STOP)
#     setMotor(CH2, 100, STOP)
#     setMotor(CH3, 100, STOP)
#     setMotor(CH4, 100, STOP)
    
# def right():
#     setMotor(CH1, 100, BACKWARD)
#     setMotor(CH2, 100, FORWARD)
#     setMotor(CH3, 100, FORWARD)
#     setMotor(CH4, 100, BACKWARD)
#     sleep(1)
#     setMotor(CH1, 0, BACKWARD)
#     setMotor(CH2, 0, FORWARD)
#     setMotor(CH3, 0, FORWARD)
#     setMotor(CH4, 0, BACKWARD)
    
# def left():
#     setMotor(CH1, 80, FORWARD)
#     setMotor(CH2, 80, BACKWARD)
#     setMotor(CH3, 80, BACKWARD)
#     setMotor(CH4, 80, FORWARD)
    
# def rotate():
#     setMotor(CH1, 80, FORWARD)
#     setMotor(CH2, 80, BACKWARD)
#     setMotor(CH3, 80, FORWARD)
#     setMotor(CH4, 80, BACKWARD)

############################################################
#제어 시작

setMotor(CH1, 100, FORWARD)
setMotor(CH2, 100, FORWARD)
setMotor(CH3, 100, FORWARD)
setMotor(CH4, 100, FORWARD)
sleep(10)
setMotor(CH1, 0, FORWARD)
setMotor(CH2, 0, FORWARD)
setMotor(CH3, 0, FORWARD)
setMotor(CH4, 0, FORWARD)
sleep(1)

# setMotor(CH1, 100, FORWARD)
# setMotor(CH2, 100, FORWARD)
# setMotor(CH3, 100, FORWARD)
# setMotor(CH4, 100, FORWARD)
# sleep(5)
# setMotor(CH1, 0, FORWARD)
# setMotor(CH2, 0, FORWARD)
# setMotor(CH3, 0, FORWARD)
# setMotor(CH4, 0, FORWARD)
# sleep(5)

setMotor(CH1, 100, BACKWARD)
setMotor(CH2, 100, BACKWARD)
setMotor(CH3, 100, BACKWARD)
setMotor(CH4, 100, BACKWARD)
sleep(10)
setMotor(CH1, 0, BACKWARD)
setMotor(CH2, 0, BACKWARD)
setMotor(CH3, 0, BACKWARD)
setMotor(CH4, 0, BACKWARD)
sleep(1)

#정지 
setMotor(CH1, 100, STOP)
setMotor(CH2, 100, STOP)
setMotor(CH3, 100, STOP)
setMotor(CH4, 100, STOP)

# 종료
GPIO.cleanup()
