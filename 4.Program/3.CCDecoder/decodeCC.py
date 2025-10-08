import numpy as np
import argparse

ERASURE_VALUE = -1 #Decode Class 내 함수 이해를 돕기 위해 -1값이 아닌 변수 이름으로 넣었습니다.


class ViterbiDecoder:
    # 참고 사이트: http://www.ktword.co.kr/test/view/view.php?no=2500
    # MATLAB내 vitdec 함수를 참고하여 파이썬 형식으로 작성했습니다. MATLAB은 8진법으로 Generate Polynomial을 받기때문에
    # 해당 코드에도 8진법으로 받았습니다. 10진법으로 하길 윈허시면 하단 argparse부분 lambda 함수를 제거하시면 될겁니다.
    # !!!!!
    # NOTE!!! : 1/2 rate에서는 정상적으로 되는걸 봤는데 2/3 rate(punc included) 에서 시험은 아직 안해봤습니다...
    # 나중에 여유될때 2/3 rate로 encoding된 샘플을 가지고 실험해보겠습니다.
    def __init__(self, args):
        # === self 변수 선언 === #
        # 파일 경로
        self.ifile = args.ifile
        self.ofile = args.ofile

        # 트렐리스 파라미터
        self.K = args.K
        self.generator = np.array([args.g1, args.g2])
        self.mode = args.mode

        # 펑처링 파라미터 (argparse로 펑쳐링 값이 선언됐을경우)
        if args.punc:
            self.punc = list(map(int, args.punc.split(','))) # args에 "1,1,0,1" 형식으로 선언된 str을 잘라서 list로~
        else:
            self.punc = None

        # K값(구속장)을 기반으로 파라미터 값 선언
        self.num_states = 2 ** (self.K - 1)
        self.tb_depth = 5 * self.K
        self.trellis_tables = None

    # ====================================================================== #
    # 이하 코드는 제미나이한테 MATLAB Vitdec함수를 python형식으로 변환후 붙여넣은 함수입니다  #
    # 샘플파일로 테스트 해봤는데 돌아가긴 하지만 혹시나 문제 생긴다면 코멘트 부탁드립니다~~~~!    #
    # ====================================================================== #

    def _generate_trellis_tables(self):
        """
        인코더의 동작과 완벽히 일치하도록 상태 전이 로직을 수정한 트렐리스 생성 함수.
        """
        num_output_bits = len(self.generator)
        gen_matrix = np.array([list(map(int, bin(g)[2:].zfill(self.K))) for g in self.generator])
        next_states = np.zeros((self.num_states, 2), dtype=int)
        outputs = np.zeros((self.num_states, 2, num_output_bits), dtype=int)

        for current_state in range(self.num_states):
            for input_bit in range(2):
                # 수정된 상태 전이 로직
                next_states[current_state, input_bit] = (current_state >> 1) | (input_bit << (self.K - 2))

                # 출력 계산 로직
                full_register_val = (input_bit << (self.K - 1)) | current_state
                full_register_bits = list(map(int, bin(full_register_val)[2:].zfill(self.K)))

                output_pair = []
                for g_poly in gen_matrix:
                    output_pair.append(np.sum(g_poly * full_register_bits) % 2)
                outputs[current_state, input_bit] = output_pair

        self.trellis_tables = (next_states, outputs)

    def _depuncture_bits(self, received_signal):
        p_len = np.sum(self.punc)
        full_len = len(self.punc)
        num_blocks = len(received_signal) // p_len
        depunctured_len = num_blocks * full_len
        depunctured_signal = np.full(depunctured_len, ERASURE_VALUE, dtype=int)
        valid_indices = np.where(np.array(self.punc) == 1)[0]
        rx_ptr = 0
        for i in range(depunctured_len):
            block_idx = i % full_len
            if block_idx in valid_indices:
                depunctured_signal[i] = received_signal[rx_ptr]
                rx_ptr += 1
        return depunctured_signal

    def _viterbi_decode(self, received_signal):
        next_states, outputs = self.trellis_tables
        num_output_bits = len(self.generator)
        num_symbols = len(received_signal) // num_output_bits

        path_metrics = np.full(self.num_states, np.inf)
        path_metrics[0] = 0
        survivor_paths = np.zeros((self.num_states, num_symbols), dtype=int)

        for t in range(num_symbols):
            received_symbol = received_signal[t * num_output_bits:(t + 1) * num_output_bits]
            next_path_metrics = np.full(self.num_states, np.inf)
            for current_state in range(self.num_states):
                if path_metrics[current_state] == np.inf: continue
                for input_bit in range(2):
                    expected_output = outputs[current_state, input_bit]
                    valid_mask = (received_symbol != ERASURE_VALUE)
                    branch_metric = np.sum(received_symbol[valid_mask] != expected_output[valid_mask])
                    candidate_metric = path_metrics[current_state] + branch_metric
                    next_s = next_states[current_state, input_bit]
                    if candidate_metric < next_path_metrics[next_s]:
                        next_path_metrics[next_s] = candidate_metric
                        survivor_paths[next_s, t] = current_state
            path_metrics = next_path_metrics

        decoded_bits = np.zeros(num_symbols, dtype=np.uint8)
        if self.mode == 'term':
            last_state = 0
        else:
            last_state = np.argmin(path_metrics)
        for t in range(num_symbols - 1, -1, -1):
            previous_state = survivor_paths[last_state, t]
            if next_states[previous_state, 0] == last_state:
                decoded_bits[t] = 0
            else:
                decoded_bits[t] = 1
            last_state = previous_state
        if self.mode == 'term': decoded_bits = decoded_bits[:-(self.K - 1)]
        return decoded_bits

    def decode(self):
        """
        파일 읽기부터 디코딩, 바이너리 파일 저장까지의 전체 과정을 실행합니다.
        """
        print("--- Viterbi 디코더 실행 ---")
        print(f"입력 파일: {self.ifile}")
        print(f"K = {self.K}, Generators = {self.generator}, Mode = {self.mode}")
        if self.punc:
            print(f"Puncture Pattern: {self.punc}")

        self._generate_trellis_tables()

        try:
            byte_array = np.fromfile(self.ifile, dtype=np.uint8)
            received_bits = np.unpackbits(byte_array)  # MSB-first가 기본 (제일 왼쪽이 최고차항!)
            print(f"총 {len(received_bits)} 비트를 파일에서 읽었습니다.")
        except FileNotFoundError:
            print(f"오류: 파일을 찾을 수 없습니다 - {self.ifile}")
            return

        if self.punc: # terminal에서 punc args에 아무것도 안넣었으면 False로 들어가서 해당 if문은 스킵될겁니다.
            received_bits = self._depuncture_bits(received_bits)

        decoded_bits = self._viterbi_decode(received_bits)
        print("디코딩 완료.")

        # 디코딩된 비트 스트림을 바이트로 묶어 바이너리 파일로 저장
        packed_bytes = np.packbits(decoded_bits)
        packed_bytes.tofile(self.ofile)

        print(f"디코딩된 비트 {len(decoded_bits)}개를 '{self.ofile}' 파일로 저장했어요!.")
        print("--- Done!!! ---")


if __name__ == '__main__':
    #여기부터~
    parser = argparse.ArgumentParser(description="Class-based Viterbi Decoder for binary files."
                                                 "K는 대문자로! 생성다항식은 8진법으로 0o빼고 숫자만 넣어주세요!"
                                                 "예시: python decodeCC.py -i encoded_zero_bits_test1.bin -o test2.bin -K 4 -g1 13 -g2 5 -mode trunc")
    parser.add_argument('-ifile', required=True, type=str, help="Input file path.")
    parser.add_argument('-ofile', required=True, type=str, help="Output file path.")
    parser.add_argument('-K', required=True, type=int, help="Constraint Length (K).")
    parser.add_argument('-g1', required=True, type=lambda x: int(x, 8), help="First generator polynomial (in octal).")
    parser.add_argument('-g2', required=True, type=lambda x: int(x, 8), help="Second generator polynomial (in octal).")
    parser.add_argument('-mode', required=True, type=str, choices=['trunc', 'term'], help="Operation mode.")
    parser.add_argument('-punc', type=str, help="Puncture pattern (e.g., '1,1,1,0').")

    args = parser.parse_args()
    #~여기까지 argparse를 이용한 인자선언입니다

    decoder = ViterbiDecoder(args)
    decoder.decode()
