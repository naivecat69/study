import numpy as np
import sys
import os


# --- 1. ViterbiDecoder 클래스 정의 ---
class ViterbiDecoder:
    """컨볼루션 코드에 대한 Viterbi 디코더 (Rate 1/2)"""

    def __init__(self, K, g_polynomials_bin):
        """
        K: 구속 길이 (Constraint Length)
        g_polynomials_bin: 이진 문자열로 된 생성 다항식 리스트 (예: ['0b111', '0b101'])
        """
        self.K = K
        self.NUM_STATES = 2 ** (K - 1)
        self.RATE = len(g_polynomials_bin)  # Rate 1/RATE (여기서는 2)

        # 이진 문자열을 비트 리스트로 변환하고 K 길이로 정규화
        self.GENERATOR_POLYNOMIALS = [
            self._parse_poly(p, K) for p in g_polynomials_bin
        ]

        self.TRELLIS = self._setup_trellis()
        print(f"디코더 초기화: K={self.K}, 상태 수={self.NUM_STATES}, G={g_polynomials_bin}")

    def _parse_poly(self, poly_bin, K):
        """이진 문자열(예: '0b101')을 K 길이의 비트 리스트로 변환"""
        # '0b' 제거 후 문자열 반전 (LSB가 앞에 오도록)
        bits_str = poly_bin[2:]

        # 다항식의 길이가 K를 초과할 경우 오류 발생 가능. K 길이로 자르거나 패딩
        if len(bits_str) > K:
            raise ValueError(f"생성 다항식 길이가 K={K}를 초과합니다: {poly_bin}")

        # K 길이로 0 패딩 (앞쪽, MSB)
        bits_list = [int(b) for b in bits_str]
        padded_bits = [0] * (K - len(bits_list)) + bits_list
        return padded_bits

    def _get_output(self, state, input_bit):
        """현재 상태와 입력 비트에 대한 인코더의 출력 비트(RATE 개) 계산"""
        # 레지스터: [새 입력] + [이전 상태 K-1]
        register = [input_bit] + list(state)
        output = []

        for g in self.GENERATOR_POLYNOMIALS:
            # XOR 연산
            out_bit = 0
            for i in range(self.K):
                if g[i] == 1:
                    out_bit ^= register[i]
            output.append(out_bit)

        return tuple(output)

    def _setup_trellis(self):
        """트렐리스(상태 전이) 정보를 미리 계산"""
        trellis = {}
        for state_dec in range(self.NUM_STATES):
            # 상태를 (K-1) 길이의 튜플 (MSB: 가장 최근 입력 비트)로 표현
            current_state = tuple(map(int, format(state_dec, '0' + str(self.K - 1) + 'b')))

            trellis[current_state] = {}
            for input_bit in [0, 1]:
                # 다음 상태: 현재 상태를 오른쪽으로 쉬프트하고 새로운 입력 비트를 MSB에 추가
                next_state = current_state[1:] + (input_bit,)
                output = self._get_output(current_state, input_bit)

                # (입력 비트) -> (다음 상태, 출력 비트)
                trellis[current_state][input_bit] = (next_state, output)
        return trellis

    # --- 2. 경로 거리 계산 함수 ---

    def _branch_metric_hard(self, received, expected):
        """하드 디시전: 해밍 거리 계산 (오류 개수)"""
        return sum(b1 != b2 for b1, b2 in zip(received, expected))

    def _branch_metric_soft(self, received_y, expected_b):
        """소프트 디시전: 유클리드 거리 제곱 계산"""
        # received_y: [-1.0, 1.0, 0.5]와 같은 실수 값 (채널 출력)
        # expected_b: (0, 1) 또는 (1, 1)과 같은 이진 비트

        metric = 0.0
        for y, b in zip(received_y, expected_b):
            # 이진 비트를 BPSK 신호로 변환 (0 -> +1, 1 -> -1)
            x = 1.0 if b == 0 else -1.0
            metric += (y - x) ** 2

        return metric

    # --- 3. 파일 입출력 ---

    def read_encoded_data_bytes(self, filename):
        """바이너리 파일에서 바이트 단위로 비트 시퀀스를 읽어옴"""
        print(f"파일 읽기 시작: {filename}")
        try:
            with open(filename, 'rb') as f:
                data = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"오류: 파일을 찾을 수 없습니다: {filename}")

        bit_sequence = []
        for byte in data:
            # 각 바이트를 8비트로 변환 (MSB부터)
            for i in range(8):
                bit_sequence.append((byte >> (7 - i)) & 1)

        return bit_sequence

    def read_soft_data_floats(self, filename):
        """소프트 디시전용: 부동 소수점(실수) 데이터를 읽어옴 (별도 파일 필요)"""
        print(f"소프트 파일 읽기 시작: {filename}")
        try:
            # 여기서는 float이 4바이트라고 가정하고 np.fromfile 사용
            return np.fromfile(filename, dtype=np.float32).tolist()
        except FileNotFoundError:
            raise FileNotFoundError(f"오류: 소프트 디시전 파일을 찾을 수 없습니다: {filename}")

    # --- 4. 디코딩 핵심 함수 (Viterbi) ---

    def _viterbi_core(self, encoded_sequence, branch_metric_func):
        """하드/소프트 디코딩의 공통 로직"""

        # 인코딩된 시퀀스를 출력 비트 쌍으로 분할
        if len(encoded_sequence) % self.RATE != 0:
            print("경고: 인코딩된 시퀀스 길이가 RATE로 나누어떨어지지 않아 마지막 비트를 무시합니다.")
            encoded_sequence = encoded_sequence[:-(len(encoded_sequence) % self.RATE)]

        encoded_chunks = [tuple(encoded_sequence[i:i + self.RATE])
                          for i in range(0, len(encoded_sequence), self.RATE)]

        T = len(encoded_chunks)  # 시간 단계의 수 (원본 입력 비트의 수)

        path_metrics = np.full((T + 1, self.NUM_STATES), np.inf)
        path_metrics[0][0] = 0.0  # 시작 상태 (0 상태) 경로 0

        # pointers[t][next_state_dec]: t번째 시간에 next_state_dec로 도달하는 최적 경로의 이전 상태 (dec)
        pointers = np.zeros((T, self.NUM_STATES), dtype=int)

        def dec_to_state(dec):
            return tuple(map(int, format(dec, '0' + str(self.K - 1) + 'b')))

        # 전진 단계 (Forward Pass)
        for t in range(T):
            current_chunk = encoded_chunks[t]

            for current_state_dec in range(self.NUM_STATES):
                current_state = dec_to_state(current_state_dec)
                current_metric = path_metrics[t][current_state_dec]

                if current_metric == np.inf:
                    continue

                for input_bit in [0, 1]:
                    next_state, expected_output = self.TRELLIS[current_state][input_bit]

                    # 브랜치 거리 계산
                    branch_metric = branch_metric_func(current_chunk, expected_output)
                    new_metric = current_metric + branch_metric

                    # 경로 비교 및 업데이트 (Add-Compare-Select)
                    next_state_dec = int("".join(map(str, next_state)), 2)

                    if new_metric < path_metrics[t + 1][next_state_dec]:
                        path_metrics[t + 1][next_state_dec] = new_metric
                        pointers[t][next_state_dec] = current_state_dec

        # 역추적 단계 (Traceback)
        final_state_dec = np.argmin(path_metrics[T])
        decoded_bits = []
        current_state_dec = final_state_dec

        for t in range(T - 1, -1, -1):
            previous_state_dec = pointers[t][current_state_dec]
            # 전이를 유발한 입력 비트는 현재 상태 튜플의 마지막 비트
            input_bit = dec_to_state(current_state_dec)[-1]

            decoded_bits.append(input_bit)
            current_state_dec = previous_state_dec

        decoded_bits.reverse()
        return decoded_bits, path_metrics[T][final_state_dec]

    def decode_hard(self, filename="encoded_data.bin"):
        """하드 디시전 디코딩 (해밍 거리 사용)"""
        # 바이너리 파일에서 0/1 비트 시퀀스를 읽음
        encoded_bits = self.read_encoded_data_bytes(filename)

        # 튜플로 변환 (함수 시그니처 맞추기)
        hard_bits = [b for b in encoded_bits]

        print("\n--- 하드 디시전 디코딩 시작 ---")
        decoded_result, final_metric = self._viterbi_core(hard_bits, self._branch_metric_hard)

        print(f"최종 경로 거리 (해밍 거리): {final_metric}")
        return decoded_result

    def decode_soft(self, filename="soft_data.bin"):
        """소프트 디시전 디코딩 (유클리드 거리 제곱 사용)"""
        # 소프트 디시전 파일에서 실수 시퀀스를 읽음
        soft_data = self.read_soft_data_floats(filename)

        if len(soft_data) % self.RATE != 0:
            raise ValueError(f"소프트 디시전 데이터 길이가 RATE={self.RATE}의 배수가 아닙니다.")

        # 소프트 디코딩은 branch_metric_soft 함수가 실수 값과 이진 비트를 비교해야 함.
        # encoded_sequence를 소프트 데이터(실수)로 사용하고,
        # branch_metric_func의 인자를 조정해야 합니다.
        # 여기서는 편의를 위해 _viterbi_core가 받는 인자를 float 리스트로 간주하고,
        # branch_metric_soft 내부에서만 expected_b를 사용하도록 구현합니다.

        print("\n--- 소프트 디시전 디코딩 시작 ---")
        decoded_result, final_metric = self._viterbi_core(soft_data, self._branch_metric_soft)

        print(f"최종 경로 거리 (유클리드 거리 제곱 합): {final_metric:.2f}")
        return decoded_result


# --- 5. 실행 함수 ---
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("사용법: python viterbi_decoder.py <K> <G1_bin> <G2_bin>")
        print("예: python viterbi_decoder.py 3 0b111 0b101")
        sys.exit(1)

    try:
        K = int(sys.argv[1])
        g1_bin = sys.argv[2]
        g2_bin = sys.argv[3]

        # K 값 검증
        if K < 2:
            raise ValueError("K(구속 길이)는 2 이상이어야 합니다.")

        # ViterbiDecoder 객체 생성
        decoder = ViterbiDecoder(K, [g1_bin, g2_bin])

        # --- 디코딩 실행 ---

        # 1. 하드 디시전 디코딩
        HARD_FILE = "encoded_data.bin"
        if os.path.exists(HARD_FILE):
            hard_decoded = decoder.decode_hard(HARD_FILE)
            print("-" * 40)
            print("✅ 하드 디시전 최종 디코딩 결과:")
            print("".join(map(str, hard_decoded)))
            print("-" * 40)
        else:
            print(f"\n[건너뛰기] {HARD_FILE} 파일이 없어 하드 디코딩을 건너뜁니다.")

        # 2. 소프트 디시전 디코딩
        SOFT_FILE = "soft_data.bin"
        if os.path.exists(SOFT_FILE):
            soft_decoded = decoder.decode_soft(SOFT_FILE)
            print("-" * 40)
            print("✅ 소프트 디시전 최종 디코딩 결과:")
            print("".join(map(str, soft_decoded)))
            print("-" * 40)
        else:
            print(f"\n[건너뛰기] {SOFT_FILE} 파일이 없어 소프트 디코딩을 건너뜁니다.")

    except ValueError as e:
        print(f"\n[오류] 입력 값 문제: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n[오류] 파일 문제: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[치명적인 오류] 예상치 못한 오류 발생: {e}")
        sys.exit(1)