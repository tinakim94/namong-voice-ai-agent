
# 가상의 쇼핑몰 주문 데이터
orders = {
    "1234": {
        "status": "배송 중",
        "delivery_date": "2026-09-27"
    },
    "5678": {
        "status": "배송 완료",
        "delivery_date": "2026-09-23"
    },
    "9999": {
        "status": "상품 준비 중",
        "delivery_date": None
    }
}


# 주문번호로 주문 정보 조회
def get_order_status(order_id):

    if order_id in orders:
        return orders[order_id]

    return None


# 직접 실행했을 때만 테스트
if __name__ == "__main__":

    order_id = input("조회할 주문번호를 입력하세요: ")

    result = get_order_status(order_id)

    if result:
        print("주문 상태:", result["status"])
        print("배송 예정일:", result["delivery_date"])
    else:
        print("해당 주문번호를 찾을 수 없습니다.")