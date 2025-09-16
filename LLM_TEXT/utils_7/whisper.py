import openai
import os
from dotenv import load_dotenv

load_dotenv()


def call_whisper_api(audio_file):
    try:
        # 임시 파일로 저장
        temp_path = f"temp_audio_{audio_file.name}"
        with open(temp_path, "wb") as f:
            for chunk in audio_file.chunks():
                f.write(chunk)

        # OpenAI 1.0.0+ 방식으로 Whisper API 호출
        client = openai.OpenAI()
        with open(temp_path, "rb") as audio_file_obj:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", file=audio_file_obj, language="ko"
            )

        # 임시 파일 삭제
        os.remove(temp_path)

        return transcript.text

    except Exception as e:
        print(f"Whisper API 에러: {str(e)}")
        return f"음성 인식 중 오류가 발생했습니다: {str(e)}"
