import numpy as np
import os
import sys

# --- 컨볼루션 인코더 파라미터 ---
K = 4
GENERATOR_POLYNOMIALS = [
    [1, 0, 1, 1],  # g1(x) = x^3 + x + 1 => 1011
    [0, 1, 0, 1]  # g2(x) = x^2 + 1     => 0101 (K=4에 맞게 0 패딩)
]
RATE = 2  # Rate 1/2


# --- 1. 인코딩 함수 ---
def encode_convolutional(input_bits, K, G_polys):
    """
    주어진 입력 비트 스트림을 컨볼루션 코드로 인코딩합니다.
    (K-1)개의 종결 0-비트(Tail Bits)를 추가하여 레지스터를 0 상태로 되돌립니다.
    """

    # K-1 개의 종결 0 비트 (Tail Bits) 추가
    # 이는 인코더의 레지스터를 마지막에 모두 0 상태로 되돌려
    # 비터비 디코더가 최종 상태 0으로 끝나도록 보장합니다.
    tail_bits = [0] * (K - 1)

    # 실제 인코딩에 사용되는 입력 비트
    extended_input = input_bits + tail_bits

    # 레지스터 초기화 (모두 0)
    register = [0] * (K - 1)
    encoded_output = []

    # 인코딩 루프
    for input_bit in extended_input:
        # 레지스터의 MSB에 새로운 입력 비트를 추가합니다.
        # 레지스터 상태: [I_n, I_{n-1}, ..., I_{n-K+2}]
        full_register = [input_bit] + register

        output_pair = []
        for g in G_polys:
            # XOR 연산 (출력 비트 계산)
            out_bit = 0
            for i in range(K):
                if g[i] == 1:
                    out_bit ^= full_register[i]
            output_pair.append(out_bit)

        encoded_output.extend(output_pair)

        # 레지스터 쉬프트: 새로운 상태는 [I_{n-1}, ..., I_{n-K+2}, I_{n-K+1}]
        # 즉, 이전 레지스터를 오른쪽으로 쉬프트하고 입력 비트를 추가
        register = [input_bit] + register[:-1]

    return encoded_output


# --- 2. 파일 저장 함수 ---
def save_encoded_bits_to_bin(encoded_bits, filename):
    """
    인코딩된 비트 리스트를 바이트 단위로 패딩하여 바이너리 파일로 저장합니다.
    """
    total_bits = len(encoded_bits)
    # 8의 배수로 만들기 위해 필요한 0 패딩 비트 수 계산
    padding_len = (8 - (total_bits % 8)) % 8

    # 패딩 비트를 맨 뒤에 추가 (일반적으로 0을 사용)
    padded_bits = encoded_bits + [0] * padding_len

    byte_array = bytearray()

    # 8비트씩 묶어 바이트로 변환
    for i in range(0, len(padded_bits), 8):
        byte_str = "".join(map(str, padded_bits[i:i + 8]))
        byte_value = int(byte_str, 2)
        byte_array.append(byte_value)

    with open(filename, 'wb') as f:
        f.write(byte_array)

    print(f"✅ 파일 저장 완료: {filename}")
    print(f"   - 총 인코딩 비트 (Tail 포함): {total_bits} 비트")
    print(f"   - 패딩 비트 수: {padding_len} 비트")
    print(f"   - 총 파일 크기: {len(byte_array)} 바이트")


# --- 3. 테스트 케이스 생성 및 인코딩 ---

def create_test_files():
    print("--- 컨볼루션 인코더를 이용한 테스트 파일 생성 시작 ---")

    # == 테스트 케이스 1: 100개의 Zero Bit ==
    TEST_FILE_1 = "encoded_zero_bits_test1.bin"

    # 입력: 100개의 0 비트
    input_bits_1 = [0] * 100

    # 인코딩 수행 (종결 비트 포함)
    encoded_bits_1 = encode_convolutional(input_bits_1, K, GENERATOR_POLYNOMIALS)

    # 결과 저장
    save_encoded_bits_to_bin(encoded_bits_1, TEST_FILE_1)

    print("-" * 40)

    # == 테스트 케이스 2: "Hello World!" ASCII ==
    TEST_FILE_2 = "encoded_helloworld_test2.bin"
    ascii_string = "Hello World!"

    # 문자열을 ASCII 코드 (바이트)로 변환 후 비트 리스트로 변환
    input_bytes_2 = ascii_string.encode('ascii')
    input_bits_2 = []
    for byte in input_bytes_2:
        # 8비트 이진수로 변환 후 리스트에 추가 (MSB부터)
        bit_str = format(byte, '08b')
        input_bits_2.extend([int(b) for b in bit_str])

    # 입력 비트 수: 12 문자 * 8 비트/문자 = 96 비트
    print(f"입력 문자열: '{ascii_string}' ({len(input_bytes_2)} 바이트)")

    # 인코딩 수행
    encoded_bits_2 = encode_convolutional(input_bits_2, K, GENERATOR_POLYNOMIALS)

    # 결과 저장
    save_encoded_bits_to_bin(encoded_bits_2, TEST_FILE_2)
    print("-" * 40)

    print("생성 다항식 정보:")
    print(f"G1 이진: {''.join(map(str, GENERATOR_POLYNOMIALS[0]))}")
    print(f"G2 이진: {''.join(map(str, GENERATOR_POLYNOMIALS[1]))}")


if __name__ == "__main__":
    create_test_files()