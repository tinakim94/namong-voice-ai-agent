"""Namong Voice AI: local Silero VAD + YAMNet audio-event analysis.
Models are user-downloaded. No cloud audio API or recording retention.
"""
import csv
import re
import subprocess
import tempfile
import time
import wave
import winsound
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sounddevice as sd
from prometheus_client import Counter, Histogram, start_http_server
from agent import process_question

BASE = Path(__file__).resolve().parent
STT = BASE / 'stt'
MODEL_DIR = STT / 'audio_models'
WHISPER = STT / 'ggml-base.bin'
PIPER = STT / 'ko_KR-kss-medium.onnx'
PIPER_PYTHON = BASE / '.venv-tts' / 'Scripts' / 'python.exe'
SILERO = MODEL_DIR / 'silero_vad.onnx'
YAMNET = MODEL_DIR / 'yamnet.onnx'
CLASS_MAP = MODEL_DIR / 'yamnet_class_map.csv'
SAMPLE_RATE = 16000
FRAME_SIZE = 512  # Silero ONNX: 32 ms at 16 kHz
WAIT_SECONDS = 5.0
END_SILENCE_SECONDS = 0.85
VAD_THRESHOLD = 0.50
MIN_SPEECH_FRAMES = 4  # 128 ms; test short '네' under real mic conditions
QUESTION_LIMIT = 10.0
CONFIRM_LIMIT = 4.0
MAX_ATTEMPTS = 2
PORT = 8001

REQUESTS = Counter('voice_agent_requests_total', 'Voice requests started')
COMPLETED = Counter('voice_agent_completed_total', 'Voice requests completed')
ERRORS = Counter('voice_agent_errors_total', 'Unexpected voice-agent errors')
STT_TIME = Histogram('voice_agent_stt_duration_seconds', 'STT execution seconds')
AI_TIME = Histogram('voice_agent_ai_duration_seconds', 'AI lookup seconds')
TTS_TIME = Histogram('voice_agent_tts_duration_seconds', 'Final-answer TTS generation and playback seconds')
TOTAL_TIME = Histogram('voice_agent_total_duration_seconds', 'Completed call total seconds')
RECORD_TIME = Histogram('voice_agent_record_duration_seconds', 'Recorded audio duration in seconds', ['step'])
NOISE_TIME = Histogram('voice_agent_noise_analysis_duration_seconds', 'Audio-event classification seconds')
EVENTS = Counter('voice_agent_audio_event_windows_total', 'Classified audio windows; tentative model labels', ['category'])

DIGITS = dict(zip('영일이삼사오육칠팔구', '0123456789'))
DIGITS['공'] = '0'
DIGITS['륙'] = '6'
DIGIT_RE = r'[0-9영공일이삼사오육륙칠팔구]'
GAP_RE = r'[\s,，.\-]*'
ORDER_RE = re.compile(
    r'(?:주문\s*번호|주본\s*번호|주문\s*반호|춤은\s*번호)'
    r'\s*(?:는|은|가|:|：|#)?\s*'
    rf'((?:{DIGIT_RE}{GAP_RE}){{4,}})'
)
BARE_RE = re.compile(rf'^\s*((?:{DIGIT_RE}{GAP_RE}){{4,}})(?:요|입니다)?[.!?。！？\s]*$')
YES = {'네', '예', '넵', '맞아요', '맞습니다', '네맞아요', '예맞아요', '확인합니다'}
NO = {'아니요', '아니오', '아뇨', '아닙니다', '틀렸어요', '틀립니다', '취소합니다'}


def find_whisper():
    files = list(STT.rglob('whisper-cli.exe'))
    if not files:
        raise FileNotFoundError('stt 폴더 안에 whisper-cli.exe가 없습니다.')
    return files[0]


class SileroVad:
    """Official Silero ONNX streaming interface without torch."""
    def __init__(self, model_path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=['CPUExecutionProvider'])
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)

    def probability(self, int16_samples):
        x = int16_samples.reshape(1, -1).astype(np.float32) / 32768.0
        if x.shape != (1, FRAME_SIZE):
            raise ValueError(f'VAD expected (1, {FRAME_SIZE}), received {x.shape}')
        data = np.concatenate([self.context, x], axis=1)
        out, self.state = self.session.run(None, {
            'input': data,
            'state': self.state,
            'sr': np.array(SAMPLE_RATE, dtype=np.int64),
        })
        self.context = data[:, -64:].copy()
        return float(np.asarray(out).reshape(-1)[0])


class NoiseClassifier:
    """Offline audio-event labels; NOT denoising or source separation."""
    def __init__(self, model_path, class_csv):
        self.session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        with class_csv.open(encoding='utf-8-sig', newline='') as f:
            self.names = {int(r['index']): r['display_name'] for r in csv.DictReader(f)}

    def inspect(self, pcm):
        # Non-overlapping 1.5s windows; 16k mono waveform floats [-1,1].
        start_time = time.perf_counter()
        audio = pcm.astype(np.float32) / 32768.0
        chunk_len = int(1.5 * SAMPLE_RATE)
        windows = []
        for start in range(0, len(audio), chunk_len):
            piece = audio[start:start + chunk_len]
            if len(piece) < SAMPLE_RATE:
                piece = np.pad(piece, (0, SAMPLE_RATE - len(piece)))
            scores = np.asarray(self.session.run(None, {self.input_name: piece})[0])
            if scores.ndim != 2 or scores.shape[1] != len(self.names):
                raise ValueError(f'YAMNet score shape differs from CSV: {scores.shape}')
            avg = scores.mean(axis=0)
            top = np.argsort(avg)[-3:][::-1]
            best = [(self.names[int(i)], round(float(avg[i]), 3)) for i in top]
            # Top label is a model hypothesis, not proof of noise/speech source.
            name = best[0][0]
            if name in {'Speech', 'Conversation', 'Narration, monologue', 'Child speech, kid speaking'}:
                category = 'speech_top'
            elif name in {'Silence', 'Inside, small room', 'Quiet'}:
                category = 'quiet_top'
            else:
                category = 'other_sound_top'
            EVENTS.labels(category=category).inc()
            windows.append({'start_s': round(start / SAMPLE_RATE, 2),
                            'end_s': round(min(start + chunk_len, len(audio)) / SAMPLE_RATE, 2),
                            'top3': best, 'category': category})
        elapsed = time.perf_counter() - start_time
        NOISE_TIME.observe(elapsed)
        return windows, elapsed


def record_audio(path, vad, classifier, step, max_duration):
    vad.reset()  # Do not carry recurrent state between separate utterances.
    pre = deque(maxlen=10)
    frames = []
    started = False
    spoken = 0
    silence = 0
    waited = 0
    silence_limit = max(1, round(END_SILENCE_SECONDS * SAMPLE_RATE / FRAME_SIZE))
    wait_limit = round(WAIT_SECONDS * SAMPLE_RATE / FRAME_SIZE)
    speech_limit = round(max_duration * SAMPLE_RATE / FRAME_SIZE)
    print(f'\n[MIC/{step}] 지금 말씀해 주세요. 발화 종료 시 자동으로 녹음합니다.')
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', blocksize=FRAME_SIZE) as stream:
        while True:
            chunk, overflowed = stream.read(FRAME_SIZE)
            if overflowed:
                print('[MIC] 마이크 입력 오버플로: 일부 오디오 누락 가능')
            raw = chunk[:, 0].copy()
            score = vad.probability(raw)
            if not started:
                pre.append(raw)
                waited += 1
                if score >= VAD_THRESHOLD:
                    started = True
                    spoken = 1
                    frames.extend(pre)
                    print(f'[VAD] 발화 시작 감지 (score={score:.2f})')
                elif waited >= wait_limit:
                    print('[VAD] 시작 대기 시간 초과: 음성 없음')
                    return False
                continue
            frames.append(raw)
            if score >= VAD_THRESHOLD:
                spoken += 1
                silence = 0
            else:
                silence += 1
            if spoken >= MIN_SPEECH_FRAMES and silence >= silence_limit:
                print('[VAD] 발화 종료 감지')
                break
            if len(frames) >= speech_limit:
                print('[VAD] 최대 발화 시간 도달')
                break
    if spoken < MIN_SPEECH_FRAMES:
        print('[VAD] 유효한 음성을 충분히 감지하지 못함')
        return False
    pcm = np.concatenate(frames)
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm.tobytes())
    seconds = len(pcm) / SAMPLE_RATE
    RECORD_TIME.labels(step=step).observe(seconds)
    print(f'[MIC] 녹음 완료: {seconds:.2f}초')
    try:
        windows, elapsed = classifier.inspect(pcm)
        for w in windows:
            print(f'[NOISE] {w["start_s"]:.1f}~{w["end_s"]:.1f}s: {w["top3"]} ({w["category"]})')
        print(f'[NOISE] 오디오 이벤트 분석 {elapsed:.2f}초 (음성/소음 분리·제거 아님)')
    except Exception as exc:
        # Diagnostic classifier failure must NOT block safe order-ID confirmation.
        print(f'[NOISE] 분석 실패(상담은 계속): {exc}')
    return True


def transcribe(audio_path, prefix, whisper_exe):
    start = time.perf_counter()
    run = subprocess.run([str(whisper_exe), '-m', str(WHISPER), '-f', str(audio_path),
                          '-l', 'ko', '-otxt', '-of', str(prefix)],
                         cwd=str(whisper_exe.parent), capture_output=True, text=True,
                         encoding='utf-8', errors='replace', timeout=120)
    if run.returncode != 0:
        raise RuntimeError('Whisper 실패: ' + run.stderr[-700:])
    txt = Path(str(prefix) + '.txt')
    if not txt.is_file():
        raise FileNotFoundError('Whisper 전사 파일이 생성되지 않았습니다.')
    STT_TIME.observe(time.perf_counter() - start)
    result = txt.read_text(encoding='utf-8-sig').strip()
    print('[STT]', result)
    return result


def speak(message, filename, final_answer=False):
    start = time.perf_counter()
    run = subprocess.run([str(PIPER_PYTHON), '-m', 'piper', '-m', str(PIPER),
                          '-f', str(filename), '--', message], cwd=str(BASE),
                         capture_output=True, text=True, encoding='utf-8',
                         errors='replace', timeout=120)
    if run.returncode != 0 or not filename.is_file():
        raise RuntimeError('Piper 실패: ' + run.stderr[-700:])
    winsound.PlaySound(str(filename), winsound.SND_FILENAME)
    if final_answer:
        TTS_TIME.observe(time.perf_counter() - start)


def extract_order_id(text, retry=False):
    match = ORDER_RE.search(text)
    if match is None and retry:
        match = BARE_RE.fullmatch(text)
    if match is None:
        return None
    cleaned = re.sub(r'[\s,，.\-]', '', match.group(1))
    order_id = ''.join(DIGITS.get(ch, ch) for ch in cleaned)
    return order_id if re.fullmatch(r'[0-9]{4,}', order_id) else None


def confirm_order_id(initial_text, temp_dir, vad, classifier, whisper_exe):
    text = initial_text
    digits_spoken = dict(zip('0123456789', '영일이삼사오육칠팔구'))
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f'\n[ORDER] 주문번호 확인 {attempt}/{MAX_ATTEMPTS}: {text}')
        order_id = extract_order_id(text, retry=attempt > 1)
        if order_id:
            print('[ORDER] 인식된 번호:', order_id)
            spoken = ', '.join(digits_spoken[d] for d in order_id)
            speak(f'주문번호 {spoken}가 맞으신가요? 네 또는 아니요로 답해 주세요.',
                  temp_dir / f'confirm_prompt_{attempt}.wav')
            confirm_audio = temp_dir / f'confirm_{attempt}.wav'
            if record_audio(confirm_audio, vad, classifier, 'confirmation', CONFIRM_LIMIT):
                response = transcribe(confirm_audio, temp_dir / f'confirm_{attempt}', whisper_exe)
                response = re.sub(r'[\s.,!?。！？]', '', response)
                if response in YES:
                    print('[ORDER] 음성 확인 완료:', order_id)
                    return order_id
                if response not in NO:
                    print('[ORDER] 불명확한 확인 응답: 안전하게 조회 중단')
                    return None
                print('[ORDER] 고객이 주문번호를 부정했습니다.')
            else:
                print('[ORDER] 확인 음성이 감지되지 않아 조회 중단')
                return None
        else:
            print('[ORDER] 주문번호 추출 실패')
        if attempt == MAX_ATTEMPTS:
            break
        speak('주문번호를 다시 말씀해 주세요.', temp_dir / f'retry_prompt_{attempt}.wav')
        retry_audio = temp_dir / f'retry_{attempt}.wav'
        if not record_audio(retry_audio, vad, classifier, 'retry', QUESTION_LIMIT):
            break
        text = transcribe(retry_audio, temp_dir / f'retry_{attempt}', whisper_exe)
    print('[ORDER] 주문번호 확인 실패: 조회 취소')
    return None


def main():
    required = [WHISPER, PIPER, PIPER_PYTHON, SILERO, YAMNET, CLASS_MAP]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f'필요한 파일이 없습니다: {path}')
    whisper_exe = find_whisper()
    vad = SileroVad(SILERO)
    classifier = NoiseClassifier(YAMNET, CLASS_MAP)
    start_http_server(PORT, addr='127.0.0.1')
    print(f'[MONITOR] http://127.0.0.1:{PORT}/metrics')
    print('나몽 Voice AI | Enter=시작, 종료=끝내기')
    while True:
        try:
            command = input('\nEnter를 누르면 녹음합니다: ').strip()
            if command == '종료':
                break
            REQUESTS.inc()
            started = time.perf_counter()
            with tempfile.TemporaryDirectory() as folder:
                temp_dir = Path(folder)
                first_audio = temp_dir / 'question.wav'
                if not record_audio(first_audio, vad, classifier, 'question', QUESTION_LIMIT):
                    continue
                text = transcribe(first_audio, temp_dir / 'question', whisper_exe)
                order_id = confirm_order_id(text, temp_dir, vad, classifier, whisper_exe)
                if order_id is None:
                    continue
                ai_start = time.perf_counter()
                result = process_question(f'주문번호 {order_id}의 배송 상태와 배송 예정일을 알려주세요.')
                AI_TIME.observe(time.perf_counter() - ai_start)
                if result.get('error'):
                    raise RuntimeError(str(result['error']))
                answer = str(result['answer'])
                print('[AI]', answer)
                speak(answer, temp_dir / 'answer.wav', final_answer=True)
                COMPLETED.inc()
                elapsed = time.perf_counter() - started
                TOTAL_TIME.observe(elapsed)
                print(f'[VOICE] 상담 정상 완료. 전체 처리 시간 {elapsed:.2f}초')
        except KeyboardInterrupt:
            print('\n상담 종료')
            break
        except Exception as exc:
            ERRORS.inc()
            print('[ERROR]', exc)


if __name__ == '__main__':
    main()
