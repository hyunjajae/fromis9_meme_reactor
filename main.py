# -*- coding: utf-8 -*-
"""
fromis_9 Meme Reactor (프로미스나인 밈 리액터)
------------------------------------------------
웹캠으로 손동작 + 표정을 인식해서, 정해진 포즈가 잡히면
해당 프로미스나인 밈 영상을 옆 창에 재생하는 프로그램입니다.

기술 스택
- OpenCV        : 웹캠 입출력, 밈 영상 재생 (끝나는 시점 완벽 감지)
- ffpyplayer    : 백그라운드 오디오 전용 재생 (싱크 버그 방지)
- MediaPipe Hands : 손 21개 랜드마크 (최대 2손)
- MediaPipe FaceMesh : 코 위치 / 얼굴 가로폭 / 입 벌림 / 찡그림(눈썹·입꼬리) 계산
"""

import os
# TensorFlow / MediaPipe 로그가 너무 많아서 조용히 시킵니다. (반드시 import 전에 설정)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

import time
import threading

import cv2
import numpy as np
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
from ffpyplayer.player import MediaPlayer   # 밈 영상 오디오 재생용


# =====================================================================
# 1. 튜닝 상수 (여기 숫자만 바꾸면 인식 민감도를 조절할 수 있습니다)
# =====================================================================
MEME_DIR = "memes"          # 밈 영상이 들어있는 폴더
CAM_INDEX = 0               # 웹캠 번호 (안 잡히면 1, 2 로 바꿔보세요)

STABLE_FRAMES = 5           # 같은 포즈가 몇 프레임 연속 유지돼야 '확정'할지 (떨림 방지)

MOUTH_OPEN_RATIO = 0.55     # 입 벌림 비율 임계값 (이상이면 '입 크게 벌림' = 그만말해인제)

# --- 찡그림(날놔라집사) 판정 (민감도 추가 하향 조정) ---
CALIB_FRAMES = 30           # 시작 시 '평소 얼굴' 측정에 쓸 프레임 수 (이 동안 가만히)
FROWN_DROP = 0.80           # 평소 대비 이 비율 미만이면 '눈썹 내림' (낮을수록 빡세게 인상 써야 함)
SQUINT_DROP = 0.73          # 평소 대비 이 비율 미만이면 '눈 찡그림' (낮을수록 빡세게 인상 써야 함)

MEME_WIDTH = 480            # 밈 영상 창 가로 크기(px). 영상이 크면 이 크기로 줄여서 재생

# 한글 출력용 폰트 (윈도우 기본 '맑은 고딕')
import platform; import platform; import platform; FONT_PATH = "C:/Windows/Fonts/malgun.ttf" if platform.system() == "Windows" else "/System/Library/Fonts/Supplemental/AppleGothic.ttf" if platform.system() == "Windows" else "/System/Library/Fonts/Supplemental/AppleGothic.ttf" if platform.system() == "Windows" else "/System/Library/Fonts/Supplemental/AppleGothic.ttf"


# =====================================================================
# 2. MediaPipe 준비
# =====================================================================
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import face_mesh as mp_face
from mediapipe.python.solutions import drawing_utils as mp_draw

# 손 랜드마크(점·선) 표시 스타일
HAND_DOT_STYLE = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4)
HAND_LINE_STYLE = mp_draw.DrawingSpec(color=(255, 80, 200), thickness=2)

hands_detector = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)
face_detector = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

# 한글 폰트 로드
try:
    FONT = ImageFont.truetype(FONT_PATH, 15)
except Exception:
    print("[경고] 맑은 고딕 폰트를 못 찾았어요. 한글이 깨질 수 있습니다.")
    FONT = ImageFont.load_default()


# =====================================================================
# 3. 손/얼굴 판정용 헬퍼 함수들
# =====================================================================
FINGER_TIP_PIP = [(8, 6), (12, 10), (16, 14), (20, 18)]

def finger_is_open(hand, tip_id, pip_id):
    return hand.landmark[tip_id].y < hand.landmark[pip_id].y

def count_open_fingers(hand):
    return sum(finger_is_open(hand, tip, pip) for tip, pip in FINGER_TIP_PIP)

def is_fist(hand):
    return count_open_fingers(hand) == 0

def wrist_y(hand):
    return hand.landmark[0].y

def mouth_open_ratio(face):
    top, bottom = face.landmark[13], face.landmark[14]
    left, right = face.landmark[61], face.landmark[291]
    vertical = abs(top.y - bottom.y)
    horizontal = abs(left.x - right.x)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal

def brow_ratio(face):
    eye_l, eye_r = face.landmark[33], face.landmark[263]
    eye_w = ((eye_l.x - eye_r.x) ** 2 + (eye_l.y - eye_r.y) ** 2) ** 0.5
    if eye_w == 0:
        return 1.0
    gap_l = face.landmark[159].y - face.landmark[105].y
    gap_r = face.landmark[386].y - face.landmark[334].y
    return ((gap_l + gap_r) / 2) / eye_w

def eye_open_ratio(face):
    def one(top, bottom, w_a, w_b):
        wl = ((face.landmark[w_a].x - face.landmark[w_b].x) ** 2
              + (face.landmark[w_a].y - face.landmark[w_b].y) ** 2) ** 0.5
        if wl == 0:
            return 0.0
        return abs(face.landmark[top].y - face.landmark[bottom].y) / wl
    return (one(159, 145, 33, 133) + one(386, 374, 362, 263)) / 2

def hand_covers_lower_face(hand, face):
    """손이 얼굴 하단(입·코)을 정확히 가리는지 정밀 판정 (재채기 포즈)"""
    nose_y = face.landmark[1].y         # 코 위치
    chin_y = face.landmark[152].y       # 턱 끝 위치
    forehead_y = face.landmark[10].y    # 이마 끝 위치
    
    # 얼굴 가로 범위 계산
    cheek_l = face.landmark[234].x
    cheek_r = face.landmark[454].x
    face_left = min(cheek_l, cheek_r)
    face_right = max(cheek_l, cheek_r)
    
    # 얼굴 전체 높이를 기준으로 정밀 유효 세로 범위 설정
    face_height = abs(forehead_y - chin_y)
    upper_limit = nose_y - face_height * 0.1   # 코보다 약간 위쪽까지만
    lower_limit = chin_y + face_height * 0.15  # 턱보다 약간 아래쪽까지만 (가슴 제외)
    
    # 주요 손가락 끝(검지8, 중지12) 및 손바닥 중심(9) 중 하나가 해당 바운딩 박스 안에 들어와야 함
    for idx in (8, 12, 9):
        p = hand.landmark[idx]
        if upper_limit < p.y < lower_limit and face_left < p.x < face_right:
            return True
    return False


# =====================================================================
# 4. 밈 판정 함수
# =====================================================================
def detect_meme(face, hands, frowning):
    n = len(hands)

    if face is not None:
        nose_y = face.landmark[1].y
        mouth_ratio = mouth_open_ratio(face)
    else:
        nose_y = None
        mouth_ratio = 0.0

    # ---- 1순위: 재채기 (손이 정확히 입/코 영역에 올 때만) ----
    if face is not None and n >= 1:
        for h in hands:
            if hand_covers_lower_face(h, face):
                return "이채영_재채기"

    # ---- 2순위: 수플렉스 (양주먹 + 두 손이 머리 위로) ----------------
    if n == 2 and nose_y is not None and all(is_fist(h) for h in hands):
        if all(wrist_y(h) < nose_y for h in hands):
            return "송하영_수플렉스"

    # ---- 3순위: 자동차 (양주먹 + 두 손이 가슴 높이에서 앞으로) -------
    if n == 2 and nose_y is not None and all(is_fist(h) for h in hands):
        if all(wrist_y(h) > nose_y for h in hands):
            return "박지원_자동차를_몰거예요"

    # ---- 4순위: 그만말해인제 (입을 크게 벌림, 손과 무관) -------------
    if face is not None and mouth_ratio >= MOUTH_OPEN_RATIO:
        return "송하영_그만말해인제"

    # ---- 5순위: 날놔라집사 (손 미검출 + 찡그린 표정) ----------------
    if not hasattr(detect_meme, "frown_cnt"):
        detect_meme.frown_cnt = 0

    if n == 0 and face is not None and frowning:
        detect_meme.frown_cnt += 1
        if detect_meme.frown_cnt >= 15: 
            return "이나경_날놔라집사"
    else:
        detect_meme.frown_cnt = max(0, detect_meme.frown_cnt - 2)

    return None


# =====================================================================
# 5. 화면에 한글 텍스트 그리기 (좌상단 위치 옵션 추가)
# =====================================================================
def draw_overlay(frame, lines, position="bottom_right"):
    fh, fw = frame.shape[:2]
    line_h, margin = 22, 14

    text_w = max(int(FONT.getlength(t)) for t, _ in lines)
    pw = text_w + 4
    ph = line_h * len(lines) + 4

    # 위치 옵션에 따른 좌표 설정
    if position == "top_left":
        x1 = margin
        y1 = margin
    else:  # 기본값: 우측 하단 (bottom_right)
        x1 = max(0, fw - pw - margin)
        y1 = max(0, fh - ph - margin)

    roi = frame[y1:y1 + ph, x1:x1 + pw]
    roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(roi_pil)
    y = 0
    for text, color in lines:
        draw.text((3, y + 1), text, font=FONT, fill=(0, 0, 0))   # 그림자
        draw.text((2, y), text, font=FONT, fill=color)           # 본문
        y += line_h
    frame[y1:y1 + ph, x1:x1 + pw] = cv2.cvtColor(np.array(roi_pil), cv2.COLOR_RGB2BGR)
    return frame


# =====================================================================
# 5-2. 고성능 밈 영상 재생기 (별도 스레드)
#   - 화면(비디오): OpenCV가 프레임을 완벽히 감지하여 무한 버퍼 렉을 제거
#   - 소리(오디오): ffpyplayer가 백그라운드 오디오 전용으로 구동 (nodisp: True)
# =====================================================================
class MemePlayer:
    def __init__(self, path, max_w):
        self.max_w = max_w
        
        # 비디오는 OpenCV로 정밀 제어
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0 or np.isnan(self.fps):
            self.fps = 30.0
        self.delay = 1.0 / self.fps
        
        # 오디오만 ffpyplayer 백그라운드로 로드
        self.player = MediaPlayer(path, ff_opts={"nodisp": True})
        
        self._frame = None
        self._done = False
        self._closed = False
        self._lock = threading.Lock()
        
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._closed:
            start_time = time.perf_counter()
            
            # OpenCV 프레임 읽기
            ret, frame = self.cap.read()
            if not ret:  # 영상 파일이 완벽히 끝나면 즉시 탈출
                break
                
            h, w = frame.shape[:2]
            if w > self.max_w:
                nh = int(h * self.max_w / w)
                frame = cv2.resize(frame, (self.max_w, nh))
                
            with self._lock:
                self._frame = frame
                
            # 오리지널 비디오 속도에 맞춰 정확하게 대기
            elapsed = time.perf_counter() - start_time
            sleep_time = self.delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        self._close()
        with self._lock:
            self._done = True

    def _close(self):
        if not self._closed:
            self._closed = True
            try:
                self.cap.release()
                self.player.close_player()
            except Exception:
                pass

    def get_latest(self):
        with self._lock:
            return self._frame, self._done

    def stop(self):
        self._close()


# =====================================================================
# 6. 메인 루프
# =====================================================================
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"[오류] 웹캠({CAM_INDEX})을 열 수 없습니다. CAM_INDEX를 바꿔보세요.")
        return

    frame_count = 0
    candidate = None
    candidate_count = 0
    meme = None
    armed = True

    brow_samples = []
    eye_samples = []
    brow_baseline = None
    eye_baseline = None

    window_main = "fromis_9 Meme Reactor"
    window_meme = "MEME"
    cv2.namedWindow(window_main)

    print("프로그램 시작! (종료하려면 웹캠 창을 클릭하고 q 를 누르세요)")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[오류] 웹캠 프레임을 읽지 못했습니다.")
            break
        frame = cv2.flip(frame, 1)
        frame_count += 1

        # 항상 화면 좌측 상단에 종료 단축키 안내 띄우기
        frame = draw_overlay(frame, [("종료하려면 q 누르기", (220, 220, 220))], position="top_left")

        # -------------------------------------------------------------
        # (A) 밈 영상 재생 상태 제어
        # -------------------------------------------------------------
        if meme is not None:
            m_frame, done = meme.get_latest()
            if m_frame is not None:
                cv2.imshow(window_meme, m_frame)
            if done:
                cv2.destroyWindow(window_meme)
                meme = None
                candidate = None
                candidate_count = 0
                armed = False

            frame = draw_overlay(frame, [("재생중...", (215, 215, 215))]) # 우측 하단
            cv2.imshow(window_main, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # -------------------------------------------------------------
        # (B) 평소 인식 루프
        # -------------------------------------------------------------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_results = hands_detector.process(rgb)
        hands = hand_results.multi_hand_landmarks or []

        face_results = face_detector.process(rgb)
        face = (face_results.multi_face_landmarks[0]
                if face_results.multi_face_landmarks else None)

        cur_brow = brow_ratio(face) if face is not None else None
        cur_eye = eye_open_ratio(face) if face is not None else None

        for h in hands:
            mp_draw.draw_landmarks(
                frame, h, mp_hands.HAND_CONNECTIONS,
                HAND_DOT_STYLE, HAND_LINE_STYLE,
            )

        # -------------------------------------------------------------
        # (B-1) 표정 보정 단계
        # -------------------------------------------------------------
        if brow_baseline is None:
            if cur_brow is not None:
                brow_samples.append(cur_brow)
                eye_samples.append(cur_eye)
            if len(brow_samples) >= CALIB_FRAMES:
                brow_baseline = sum(brow_samples) / len(brow_samples)
                eye_baseline = sum(eye_samples) / len(eye_samples)
                print(f"[보정 완료] 눈썹 baseline={brow_baseline:.3f}, 눈 baseline={eye_baseline:.3f}")
            
            lines = [
                ("표정 보정중... 가만히 있어주세요", (0, 220, 255)),
                (f"진행 {len(brow_samples)}/{CALIB_FRAMES}", (200, 200, 200)),
            ]
            frame = draw_overlay(frame, lines)
            cv2.imshow(window_main, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # -------------------------------------------------------------
        # (B-2) 찡그림 판정 및 밈 매칭
        # -------------------------------------------------------------
        if cur_brow is not None:
            brow_down = cur_brow < brow_baseline * FROWN_DROP
            squint = cur_eye < eye_baseline * SQUINT_DROP
            frowning = brow_down or squint
        else:
            frowning = False

        detected = detect_meme(face, hands, frowning)

        if detected is None:
            armed = True

        if detected == candidate:
            candidate_count += 1
        else:
            candidate = detected
            candidate_count = 1

        confirmed = candidate is not None and candidate_count >= STABLE_FRAMES
        if confirmed and armed:
            path = os.path.join(MEME_DIR, candidate + ".mp4")
            if os.path.exists(path):
                try:
                    meme = MemePlayer(path, MEME_WIDTH)
                    armed = False
                    cv2.namedWindow(window_meme)
                    cv2.moveWindow(window_main, 0, 0)
                    cv2.moveWindow(window_meme, frame.shape[1] + 20, 0)
                    print(f"[밈 재생] {candidate}")
                except Exception as e:
                    print(f"[오류] 영상을 열 수 없습니다: {path}  ({e})")
            else:
                print(f"[안내] 영상 파일이 없습니다: {path}  (memes 폴더에 넣어주세요)")
                armed = False

        # 오버레이 정보 표시
        expr = "찡그림" if frowning else "보통"
        lines = [
            (f"표정  {expr}", (255, 130, 130) if frowning else (190, 255, 190)),
            (f"동작  {detected or '-'}", (130, 220, 255)),
        ]
        frame = draw_overlay(frame, lines)
        cv2.imshow(window_main, frame)

        if frame_count % 30 == 0 and cur_brow is not None:
            print(f"[튜닝] 눈썹 {cur_brow/brow_baseline:.2f}/{FROWN_DROP}  "
                  f"눈 {cur_eye/eye_baseline:.2f}/{SQUINT_DROP}  "
                  f"(이 값이 뒤 숫자보다 작아지면 찡그림 인식)")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if meme is not None:
        meme.stop()
    cv2.destroyAllWindows()
    print("프로그램을 종료했습니다.")


if __name__ == "__main__":
    main()

