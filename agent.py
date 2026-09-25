
import json
import re
import urllib.request
import time

from orders import get_order_status


# ==========================================
# 1. 기본 설정
# ==========================================

MODEL = "qwen3:1.7b"

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"


# ==========================================
# 2. AI가 사용할 도구 정의
# ==========================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": (
                "가상 쇼핑몰의 주문번호로 "
                "주문 상태와 배송 예정일을 조회합니다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "고객이 입력한 주문번호"
                    }
                },
                "required": ["order_id"]
            }
        }
    }
]


# ==========================================
# 3. 고객 질문에서 주문번호 추출
# ==========================================

def extract_order_id(question):

    match = re.search(
        r"주문\s*번호\s*(?:는|가|:|#)?\s*(\d{4,})",
        question
    )

    if match:
        return match.group(1)

    return None


# ==========================================
# 4. Ollama AI 호출
# ==========================================

def call_ai(question):

    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "당신은 가상 쇼핑몰의 주문 조회 상담원입니다. "
                    "고객이 주문번호를 제공하면 반드시 "
                    "get_order_status 도구를 호출하세요. "
                    "도구 호출 형식을 일반 텍스트로 "
                    "출력하지 마세요. "
                    "주문 상태나 배송 날짜를 추측하지 마세요."
                )
            },
            {
                "role": "user",
                "content": question
            }
        ],
        "tools": tools,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0
        }
    }

    request = urllib.request.Request(
        url=OLLAMA_URL,
        data=json.dumps(data).encode("utf-8"),
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    with urllib.request.urlopen(
        request,
        timeout=120
    ) as response:

        result = json.load(response)

    return result["message"]


# ==========================================
# 5. 실제 주문 데이터로 답변 생성
# ==========================================

def make_answer(order_id, order_data):

    if order_data is None:

        return (
            f"주문번호 {order_id}를 "
            "찾을 수 없습니다."
        )

    status = order_data["status"]

    delivery_date = order_data["delivery_date"]

    answer = (
        f"주문번호 {order_id}는 "
        f"현재 {status} 상태입니다."
    )

    if delivery_date is None:

        answer += (
            " 배송 예정일은 아직 정해지지 않았습니다."
        )

    else:

        answer += (
            f" 배송 예정일은 {delivery_date}입니다."
        )

    return answer


# ==========================================
# 6. AI Agent 질문 처리 함수
# ==========================================

def process_question(question):

    # 질문에 있는 주문번호 확인
    expected_order_id = extract_order_id(question)

    # 주문번호가 없는 경우
    if expected_order_id is None:

        return {
            "question": question,
            "order_id": None,
            "routing": "no_order_id",
            "order_data": None,
            "answer": "조회할 주문번호를 입력해 주세요.",
            "error": None
        }

    try:

        # AI에게 질문 전달
        ai_message = call_ai(question)

        # AI가 요청한 도구 호출 확인
        tool_calls = ai_message.get(
            "tool_calls", []
        ) or []

        selected_order_id = None

        for tool_call in tool_calls:

            function_info = tool_call.get(
                "function", {}
            )

            if function_info.get(
                "name"
            ) != "get_order_status":

                continue

            arguments = function_info.get(
                "arguments", {}
            )

            # 도구 인자가 문자열이면 JSON 변환
            if isinstance(arguments, str):

                try:

                    arguments = json.loads(arguments)

                except json.JSONDecodeError:

                    continue

            if not isinstance(arguments, dict):
                continue

            order_id = str(
                arguments.get("order_id", "")
            ).strip()

            # 고객이 입력한 주문번호와
            # AI가 선택한 주문번호 비교
            if order_id == expected_order_id:

                selected_order_id = order_id

                break

        # ==================================
        # 정식 Tool Calling 성공
        # ==================================

        if selected_order_id is not None:

            routing = "tool"

            order_id = selected_order_id

        # ==================================
        # Tool Calling 실패 시 Fallback
        # ==================================

        else:

            routing = "fallback"

            order_id = expected_order_id

        # 실제 주문 조회 함수 실행
        order_data = get_order_status(order_id)

        # 실제 데이터로 답변 생성
        answer = make_answer(
            order_id,
            order_data
        )

        return {
            "question": question,
            "order_id": order_id,
            "routing": routing,
            "order_data": order_data,
            "answer": answer,
            "error": None
        }

    except Exception as error:

        return {
            "question": question,
            "order_id": expected_order_id,
            "routing": "error",
            "order_data": None,
            "answer": "주문 조회 중 오류가 발생했습니다.",
            "error": str(error)
        }


# ==========================================
# 7. 직접 실행할 때만 상담 프로그램 시작
# ==========================================

if __name__ == "__main__":

    print("=== 나몽 AI 주문 상담원 ===")

    print("종료하려면 '종료'라고 입력하세요.")

    while True:

        question = input(
            "\n고객님의 질문을 입력하세요: "
        ).strip()

        if question == "종료":

            print("상담을 종료합니다.")

            break

        if not question:

            print("질문을 입력해 주세요.")

            continue

        print("AI가 요청을 분석하고 있습니다...")

        start_time = time.perf_counter()

        result = process_question(question)

        elapsed = time.perf_counter() - start_time

        print(
            "[ROUTING]",
            result["routing"]
        )

        if result["routing"] == "tool":

            print(
                "[TOOL] 정식 Tool Calling 성공"
            )

        elif result["routing"] == "fallback":

            print(
                "[TOOL] Fallback 경로로 조회"
            )

        if result["error"]:

            print(
                "[ERROR]",
                result["error"]
            )

        print(
            "\nAI 상담원:",
            result["answer"]
        )

        print(
            f"[MONITOR] 응답 시간: {elapsed:.2f}초"
        )