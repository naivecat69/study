#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DL.test.py (정규화 + 커리큘럼 Stage1/2 통합 패치판)

기능 요약
- 합성 CW 학습 (--train_synth)
- 라벨 CSV 파인튜닝 (--finetune_csv)
- 추론 (--audio_in)
- 폴더 의사라벨 생성 (--make_pseudo)
- 정규화 파이프라인 (--normalize, --heterodyne, --bp_pre, --tnorm)
- 커리큘럼:
  * --stage s1 : 토큰(DOT/DASH/SEP_ELEM/SEP_CHAR/SEP_WORD) CTC 학습/추론
  * --stage s2 : s1의 예측 토큰열을 룩업으로 문자로 치환하여 출력(후처리)

필요 패키지:
  pip install torch torchaudio librosa soundfile numpy scipy
"""

import os, math, random, argparse, string, glob, csv
from dataclasses import dataclass, replace
from typing import List, Tuple
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, lfilter, welch

# 고정 시드
SEED = 2025
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ========================= 기본 설정 =========================
@dataclass
class Config:
    # 오디오/특징
    sr: int = 16000
    n_mels: int = 64
    hop_ms: float = 10.0
    win_ms: float = 25.0
    fmin: int = 40
    fmax: int = 4000
    pre_emph: float = 0.0
    # 정규화 플래그
    normalize: bool = True        # 전처리 파이프라인 on/off
    heterodyne: bool = False      # f0 하향변환(헤테로다인) 사용
    bp_pre: bool = True           # 좁은 대역통과 사용
    bp_bw_hz: int = 150
    tnorm: bool = False           # 속도 정규화(dit 길이 표준화)
    # 합성
    synth_wpm_min: int = 14
    synth_wpm_max: int = 28
    synth_f0_min: int = 300
    synth_f0_max: int = 1200
    atk_ms: float = 5.0
    dcy_ms: float = 5.0
    snr_db_min: float = 10.0
    snr_db_max: float = 35.0
    # 학습
    batch_size: int = 16
    epochs: int = 10
    lr: float = 3e-4
    # 모델
    hidden: int = 128
    cnn_ch: int = 64

# ========================= 모스 테이블/사전 =========================
MORSE = {
    'A':'.-','B':'-...','C':'-.-.','D':'-..','E':'.','F':'..-.','G':'--.','H':'....','I':'..',
    'J':'.---','K':'-.-','L':'.-..','M':'--','N':'-.','O':'---','P':'.--.','Q':'--.-','R':'.-.',
    'S':'...','T':'-','U':'..-','V':'...-','W':'.--','X':'-..-','Y':'-.--','Z':'--..',
    '0':'-----','1':'.----','2':'..---','3':'...--','4':'....-','5':'.....',
    '6':'-....','7':'--...','8':'---..','9':'----.',
    '.':'.-.-.-', ',':'--..--', '?':'..--..', '/':'-..-.', '-':'-....-',
    ' ':' '
}
# 문자 인식용 알파벳
VOCAB_CHARS = list(string.ascii_uppercase + string.digits + " .,?/-")

# Stage1 토큰 집합
S1_TOKENS = ["DOT","DASH","SEP_ELEM","SEP_CHAR","SEP_WORD"]  # + BLANK

# -------------------- 텍스트↔토큰 변환 --------------------
def text_to_s1_tokens(text: str) -> List[str]:
    """문자열(공백 포함)을 Stage1 토큰열로 변환"""
    toks = []
    for idx, ch in enumerate(text.upper()):
        if ch == ' ':
            toks.append("SEP_WORD")
            continue
        code = MORSE.get(ch)
        if not code:
            continue
        for i, s in enumerate(code):
            toks.append("DOT" if s=='.' else "DASH")
            if i < len(code)-1:
                toks.append("SEP_ELEM")
        # 문자 끝 간격(단어 사이 공백이 아니면)
        # 다음 문자가 공백이거나 문장의 끝이면 SEP_WORD/무시, 아니면 SEP_CHAR
        if idx < len(text)-1 and text[idx+1] != ' ':
            toks.append("SEP_CHAR")
        elif idx < len(text)-1 and text[idx+1] == ' ':
            # 다음 루프에서 SEP_WORD가 들어가므로 생략
            pass
    # 연속 SEP 제거/정리
    cleaned=[]
    prev=None
    for t in toks:
        if t.startswith("SEP") and prev and prev.startswith("SEP"):
            if prev=="SEP_WORD":  # 단어 구분이 최우선
                continue
        cleaned.append(t); prev=t
    return cleaned

def s1_tokens_to_text(tokens: List[str]) -> str:
    """Stage1 토큰열을 문자로 복원"""
    # 토큰을 dot/dash 시퀀스로 누적하다가 SEP_CHAR/SEP_WORD에서 커밋
    inv = {v:k for k,v in MORSE.items() if k!=' '}
    out=[]
    cur=[]
    for t in tokens:
        if t == "DOT": cur.append('.')
        elif t == "DASH": cur.append('-')
        elif t == "SEP_ELEM":
            # 같은 문자 내부 구분자 → 아무 것도 안 함
            pass
        elif t == "SEP_CHAR":
            code=''.join(cur); cur=[]
            if code in inv: out.append(inv[code])
        elif t == "SEP_WORD":
            code=''.join(cur); cur=[]
            if code in inv: out.append(inv[code])
            out.append(' ')
    # 잔여 커밋
    if cur:
        code=''.join(cur)
        if code in inv: out.append(inv[code])
    # 공백 정리
    s=''.join(out)
    return ' '.join(s.split())

# ========================= 합성/전처리/특징 =========================
def synth_morse(text: str, fs: int, f0: float, wpm: float,
                atk: float=0.005, dcy: float=0.005, amp: float=0.9) -> np.ndarray:
    T = 1.2 / wpm
    def tone(dur):
        n = int(round(dur*fs)); t = np.arange(n)/fs
        return np.sin(2*np.pi*f0*t)
    def ad_env(n):
        a = int(max(1, round(atk*fs))); d = int(max(1, round(dcy*fs)))
        core = max(1, n-a-d)
        return np.concatenate([np.linspace(0,1,a,endpoint=False),
                               np.ones(core),
                               np.linspace(1,0,d,endpoint=True)])[:n]
    out=[]
    for ch in text.upper():
        if ch==' ':
            out.append(np.zeros(int(round(7*T*fs)))); continue
        code = MORSE.get(ch)
        if not code: continue
        for s in code:
            dur = T if s=='.' else 3*T
            seg = tone(dur)*ad_env(int(round(dur*fs)))*amp
            out.append(seg); out.append(np.zeros(int(round(T*fs))))
        out.append(np.zeros(int(round(2*T*fs))))
    return (np.concatenate(out) if out else np.zeros(0)).astype(np.float32)

def add_awgn(x: np.ndarray, snr_db: float) -> np.ndarray:
    if len(x)==0: return x
    ps = np.mean(x**2)+1e-12
    pn = ps / (10**(snr_db/10))
    n = np.sqrt(pn)*np.random.randn(*x.shape)
    return (x+n).astype(np.float32)

def pre_emphasis(x: np.ndarray, pre: float) -> np.ndarray:
    if pre<=0: return x
    y = np.copy(x); y[1:] = x[1:] - pre*x[:-1]; return y

def welch_peak_hz(x: np.ndarray, sr: int, fmin=80, fmax=4000) -> float:
    f,P = welch(x, fs=sr, nperseg=4096)
    m = (f>=fmin)&(f<=min(fmax, sr/2-100))
    return float(f[m][np.argmax(P[m])]) if np.any(m) else 600.0

def butter_bandpass(lowcut, highcut, fs, order=4):
    return butter(order, [lowcut/(fs/2), highcut/(fs/2)], btype='band')

def apply_bandpass(x: np.ndarray, sr: int, center: float, bw_hz: float) -> np.ndarray:
    low = max(20.0, center-bw_hz/2)
    high = min(sr/2-50.0, center+bw_hz/2)
    if low>=high: return x
    b,a = butter_bandpass(low, high, sr, order=4)
    return lfilter(b,a,x).astype(np.float32)

def heterodyne_env(x: np.ndarray, sr: int, f0: float, lp_ms: float=8.0) -> np.ndarray:
    """x * e^{-j2π f0 t}의 절대값을 저역평활 → 포락선"""
    t = np.arange(len(x))/sr
    cos = np.cos(2*np.pi*f0*t).astype(np.float32)
    sin = np.sin(2*np.pi*f0*t).astype(np.float32)
    i = x * cos
    q = x * (-sin)
    env = np.sqrt(i*i + q*q)
    n_lp = max(1, int(sr*(lp_ms/1000.0)))
    k = np.ones(n_lp, dtype=np.float32)/n_lp
    return np.convolve(env, k, mode='same').astype(np.float32)

def simple_envelope_after_bpf(x: np.ndarray, sr: int, f0: float, bw_hz:int=150, lp_ms:float=8.0)->np.ndarray:
    y = apply_bandpass(x, sr, f0, bw_hz)
    y = np.abs(y)
    n_lp=max(1, int(sr*(lp_ms/1000.0)))
    k=np.ones(n_lp, dtype=np.float32)/n_lp
    return np.convolve(y, k, mode='same').astype(np.float32)

def time_normalize_envelope(env: np.ndarray, sr: int, target_T_ms: float=50.0) -> np.ndarray:
    """dit 길이 추정 후 시간축을 표준 dit 길이에 맞춰 리샘플"""
    # 적응 임계값
    thr = 0.5*(np.median(env) + 0.3*np.max(env))
    b = (env > thr).astype(np.int8)
    if len(b)==0: return env
    runs=[]; cur=b[0]; cnt=1
    for v in b[1:]:
        if v==cur: cnt+=1
        else: runs.append((cur,cnt)); cur=v; cnt=1
    runs.append((cur,cnt))
    on = [c for v,c in runs if v==1]
    if not on: return env
    T = max(1, int(np.median(on)))              # 샘플 수
    T_ms = 1000.0 * T / sr
    alpha = (target_T_ms / T_ms) if T_ms>0 else 1.0
    target_sr = int(sr * alpha)
    if target_sr < 2000: target_sr = 2000       # 너무 낮아지지 않게
    out = librosa.resample(env, orig_sr=sr, target_sr=target_sr)
    # 다시 원래 sr로 리샘플(길이만 상대 표준화 효과 유지)
    out2 = librosa.resample(out, orig_sr=target_sr, target_sr=sr)
    return out2.astype(np.float32)

def wav_to_logmel(wav: np.ndarray, sr: int, n_mels: int, hop_ms: float, win_ms: float,
                  fmin: int, fmax: int) -> np.ndarray:
    hop = int(round(sr*hop_ms/1000))
    win = int(round(sr*win_ms/1000))
    n_fft = max(512, 2**int(math.ceil(math.log2(max(256, win)))))
    S = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft,
                                       hop_length=hop, win_length=win,
                                       n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0)
    logmel = np.log1p(S).T
    m = np.mean(logmel,0,keepdims=True); s = np.std(logmel,0,keepdims=True)+1e-6
    return ((logmel-m)/s).astype(np.float32)

def load_audio_any(path: str, sr: int) -> np.ndarray:
    wav,_ = librosa.load(path, sr=sr, mono=True)
    return wav.astype(np.float32)

# ========================= 사전/모델 =========================
class Vocab:
    def __init__(self, tokens: List[str]):
        self.chars = tokens[:]
        self.blank_id = len(self.chars)
        self.stoi = {c:i for i,c in enumerate(self.chars)}
        self.itos = {i:c for i,c in enumerate(self.chars)}
        self.itos[self.blank_id] = ''  # CTC blank
    def text2ids(self, seq: List[str]) -> List[int]:
        return [self.stoi[c] for c in seq if c in self.stoi]
    def ids2seq_ctc(self, ids: List[int]) -> List[str]:
        out, prev = [], None
        for i in ids:
            if i==self.blank_id: prev=i; continue
            tok = self.itos.get(i,'')
            if i!=prev: out.append(tok)
            prev=i
        return out

class CWCTC(nn.Module):
    def __init__(self, n_mels=64, n_class=38, hidden=128, cnn_ch=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_ch, 3, padding=1), nn.ReLU(), nn.MaxPool2d((1,2)),
            nn.Conv2d(cnn_ch, cnn_ch, 3, padding=1), nn.ReLU(), nn.MaxPool2d((1,2))
        )
        proj_in = cnn_ch * (n_mels//4)
        self.proj = nn.Linear(proj_in, hidden)
        self.rnn  = nn.LSTM(hidden, hidden, num_layers=2, bidirectional=True, batch_first=True)
        self.out  = nn.Linear(hidden*2, n_class)
    def forward(self, x):            # x: (B,1,T,M)
        z = self.cnn(x)              # (B,C,T,M')
        z = z.permute(0,2,1,3)       # (B,T,C,M')
        B,T,C,Mp = z.shape
        z = z.reshape(B,T,C*Mp)      # (B,T,C*M')
        z = self.proj(z)             # (B,T,H)
        z,_ = self.rnn(z)            # (B,T,2H)
        return self.out(z)           # (B,T,C)

# ========================= 데이터셋/콜레이트 =========================
def normalize_pipeline(wav: np.ndarray, cfg: Config) -> Tuple[np.ndarray, float]:
    """정규화: f0 추정 → (헤테로다인 or BPF) → 레벨 → (선택) 속도 정규화"""
    if not cfg.normalize:
        mx=np.max(np.abs(wav))+1e-9
        return (wav/mx).astype(np.float32), 0.0
    f0 = welch_peak_hz(wav, cfg.sr, fmin=cfg.fmin, fmax=cfg.fmax)
    if cfg.heterodyne:
        env = heterodyne_env(wav, cfg.sr, f0)
    else:
        env = simple_envelope_after_bpf(wav, cfg.sr, f0, bw_hz=cfg.bp_bw_hz)
    if cfg.tnorm:
        env = time_normalize_envelope(env, cfg.sr, target_T_ms=50.0)
    # 멜 특징은 env 대신 원파형 기반도 가능하지만, 여기선 env를 표준화 입력으로 사용
    mx = np.max(np.abs(env)) + 1e-9
    return (env/mx).astype(np.float32), f0

class SynthDatasetS1(Dataset):
    """Stage1 토큰 라벨 합성 데이터셋"""
    def __init__(self, cfg: Config, vocab: Vocab, n_items=2000, min_len=10, max_len=40):
        self.cfg=cfg; self.vocab=vocab; self.n=n_items; self.min_len=min_len; self.max_len=max_len
        self.alpha = VOCAB_CHARS
    def _rand_text(self)->str:
        L = random.randint(self.min_len, self.max_len)
        chars=[]
        for _ in range(L):
            if random.random()<0.16: chars.append(' ')
            else: chars.append(random.choice(self.alpha))
        return ''.join(chars).strip()
    def __len__(self): return self.n
    def __getitem__(self, idx):
        txt = self._rand_text()
        f0  = random.uniform(self.cfg.synth_f0_min, self.cfg.synth_f0_max)
        wpm = random.uniform(self.cfg.synth_wpm_min, self.cfg.synth_wpm_max)
        wav = synth_morse(txt, fs=self.cfg.sr, f0=f0, wpm=wpm,
                          atk=self.cfg.atk_ms/1000.0, dcy=self.cfg.dcy_ms/1000.0)
        # 노이즈/드리프트
        if len(wav)>0 and random.random()<0.5:
            drift = 1.0 + np.linspace(0, random.uniform(-0.002,0.002), len(wav))
            wav = (wav*drift).astype(np.float32)
        snr = random.uniform(self.cfg.snr_db_min, self.cfg.snr_db_max)
        wav = add_awgn(wav, snr)
        wav = pre_emphasis(wav, self.cfg.pre_emph)

        # 정규화 파이프라인
        x_env, _ = normalize_pipeline(wav, self.cfg)

        # 얇은 멜(또는 env를 멜처럼 취급) – 간단화: 멜 생성에 env 사용
        mel = wav_to_logmel(x_env, self.cfg.sr, self.cfg.n_mels, self.cfg.hop_ms, self.cfg.win_ms,
                            self.cfg.fmin, self.cfg.fmax)

        # Stage1 라벨(토큰)
        tokens = text_to_s1_tokens(txt)
        target = self.vocab.text2ids(tokens)
        return mel, np.array(target, np.int32), tokens

class LabeledAudioDatasetS1(Dataset):
    """CSV(path,text) → Stage1 토큰 라벨로 변환 후 사용"""
    def __init__(self, csv_path: str, cfg: Config, vocab: Vocab, shuffle=True):
        self.cfg=cfg; self.vocab=vocab
        rows=[]
        with open(csv_path, 'r', encoding='utf-8') as f:
            r = csv.reader(f)
            for row in r:
                if not row: continue
                if row[0].lower() in ('path','file','audio') and len(row)>=2: continue
                p = row[0].strip()
                t = ','.join(row[1:]).strip() if len(row)>1 else ''
                rows.append((p,t))
        if shuffle: random.shuffle(rows)
        self.rows=rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        p, raw = self.rows[idx]
        wav = load_audio_any(p, sr=self.cfg.sr)
        wav = pre_emphasis(wav, self.cfg.pre_emph)
        x_env, _ = normalize_pipeline(wav, self.cfg)
        mel = wav_to_logmel(x_env, self.cfg.sr, self.cfg.n_mels, self.cfg.hop_ms, self.cfg.win_ms,
                            self.cfg.fmin, self.cfg.fmax)
        tokens = text_to_s1_tokens(raw.upper())
        target = self.vocab.text2ids(tokens)
        return mel, np.array(target, np.int32), tokens

def collate_ctc(batch, blank_id: int):
    T_list=[b[0].shape[0] for b in batch]; M=batch[0][0].shape[1]; T_max=max(T_list)
    mels=np.zeros((len(batch),T_max,M),np.float32)
    in_lens=[]; targets=[]; tgt_lens=[]; texts=[]
    for i,(mel,tgt,txt) in enumerate(batch):
        t=mel.shape[0]; mels[i,:t,:]=mel
        in_lens.append(t); targets.append(tgt); tgt_lens.append(len(tgt)); texts.append(txt)
    flat_targets = np.concatenate(targets) if targets else np.zeros((0,),np.int32)
    mels = torch.from_numpy(mels).unsqueeze(1)
    flat_targets = torch.from_numpy(flat_targets).long()
    in_lens = torch.tensor(in_lens, dtype=torch.int32)
    tgt_lens= torch.tensor(tgt_lens,dtype=torch.int32)
    return mels, flat_targets, in_lens, tgt_lens, texts

# ========================= 학습/파인튜닝/추론 =========================
def train_s1(cfg: Config, save_ckpt: str):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab=Vocab(S1_TOKENS); n_class=len(vocab.chars)+1
    model=CWCTC(cfg.n_mels,n_class,cfg.hidden,cfg.cnn_ch).to(device)
    ds=SynthDatasetS1(cfg,vocab,n_items=2000)
    dl=DataLoader(ds,batch_size=cfg.batch_size,shuffle=True,
                  collate_fn=lambda b: collate_ctc(b, vocab.blank_id), num_workers=0)
    crit=nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)
    optim=torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    for ep in range(1,cfg.epochs+1):
        model.train(); tot=0.0
        for mels,targets,in_lens,tgt_lens,_ in dl:
            mels=mels.to(device); targets=targets.to(device)
            in_lens=in_lens.to(device); tgt_lens=tgt_lens.to(device)
            logp = model(mels).log_softmax(-1).permute(1,0,2).contiguous()
            loss=crit(logp, targets, in_lens, tgt_lens)
            optim.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.0); optim.step()
            tot+=float(loss.item())
        print(f"[S1 Epoch {ep}/{cfg.epochs}] CTC loss={tot/max(1,len(dl)):.4f}")
    os.makedirs(os.path.dirname(save_ckpt) or ".", exist_ok=True)
    torch.save({"model":model.state_dict(),"cfg":cfg.__dict__,"vocab":vocab.chars, "stage":"s1"}, save_ckpt)
    print(f"✅ Saved checkpoint(S1): {save_ckpt}")

def finetune_s1(cfg: Config, ckpt_in: str, csv_path: str, save_ckpt: str,
                epochs=5, lr=1e-4, batch_size=8, freeze_cnn=False, freeze_rnn=False):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt=torch.load(ckpt_in, map_location=device)
    vocab=Vocab(ckpt.get("vocab", S1_TOKENS))
    n_class=len(vocab.chars)+1
    model=CWCTC(cfg.n_mels,n_class,cfg.hidden,cfg.cnn_ch).to(device)
    model.load_state_dict(ckpt["model"])
    if freeze_cnn: [setattr(p,'requires_grad',False) for p in model.cnn.parameters()]
    if freeze_rnn: [setattr(p,'requires_grad',False) for p in model.rnn.parameters()]
    ds=LabeledAudioDatasetS1(csv_path, cfg, vocab, shuffle=True)
    dl=DataLoader(ds,batch_size=batch_size,shuffle=True,
                  collate_fn=lambda b: collate_ctc(b, vocab.blank_id), num_workers=0)
    crit=nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)
    optim=torch.optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-6)
    best=float('inf')
    for ep in range(1,epochs+1):
        model.train(); tot=0.0
        for mels,targets,in_lens,tgt_lens,_ in dl:
            mels=mels.to(device); targets=targets.to(device)
            in_lens=in_lens.to(device); tgt_lens=tgt_lens.to(device)
            logp=model(mels).log_softmax(-1).permute(1,0,2).contiguous()
            loss=crit(logp, targets, in_lens, tgt_lens)
            optim.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.0); optim.step()
            tot+=float(loss.item())
        avg=tot/max(1,len(dl)); print(f"[S1 FT {ep}/{epochs}] loss={avg:.4f}")
        if avg<best:
            best=avg
            torch.save({"model":model.state_dict(),"cfg":cfg.__dict__,"vocab":vocab.chars, "stage":"s1"}, save_ckpt)
            print(f"  ↳ ✅ improved; saved: {save_ckpt}")

def infer_s1_to_tokens(cfg: Config, ckpt_path: str, audio_path: str) -> List[str]:
    """오디오 → Stage1 토큰열"""
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt=torch.load(ckpt_path, map_location=device)
    vocab=Vocab(ckpt.get("vocab", S1_TOKENS))
    n_class=len(vocab.chars)+1
    model=CWCTC(cfg.n_mels,n_class,cfg.hidden,cfg.cnn_ch).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    wav=load_audio_any(audio_path, sr=cfg.sr)
    wav=pre_emphasis(wav, cfg.pre_emph)
    x_env, f0 = normalize_pipeline(wav, cfg)
    mel=wav_to_logmel(x_env,cfg.sr,cfg.n_mels,cfg.hop_ms,cfg.win_ms,cfg.fmin,cfg.fmax)
    with torch.no_grad():
        x=torch.from_numpy(mel).unsqueeze(0).unsqueeze(1).to(device)
        logits=model(x)
        pred=logits.argmax(-1).squeeze(0).tolist()
        tokens=vocab.ids2seq_ctc(pred)
    return tokens

# ----- DSP 백업 디코더 (참고용) -----
def dsp_morse_decode(wav: np.ndarray, sr: int, fmin=100, fmax=3000):
    f0=welch_peak_hz(wav,sr,fmin=fmin,fmax=fmax)
    x=apply_bandpass(wav,sr,center=f0,bw_hz=120); x=np.abs(x)
    n_lp=max(1,int(0.008*sr)); env=np.convolve(x, np.ones(n_lp)/n_lp, mode='same')
    thr=0.5*(np.median(env)+np.max(env)*0.3); b=(env>thr).astype(np.int8)
    runs=[]; cur=b[0]; cnt=1
    for v in b[1:]:
        if v==cur: cnt+=1
        else: runs.append((cur,cnt)); cur=v; cnt=1
    runs.append((cur,cnt))
    on=[c for v,c in runs if v==1]; off=[c for v,c in runs if v==0]
    if not on or not off: return ""
    T=max(1,int(np.median(on))); sym=[]
    for v,c in runs:
        if v==1: sym.append('.' if c<2.2*T else '-')
        else:
            if c<1.5*T: pass
            elif c<4.5*T: sym.append(' ')
            else: sym.append('   ')
    inv={v:k for k,v in MORSE.items() if k!=' '}
    out=[]; buf=[]
    for s in sym:
        if s in (' ','   '):
            code=''.join(buf); buf=[]
            if code in inv: out.append(inv[code])
            if s=='   ': out.append(' ')
        else:
            buf.append(s)
    if buf:
        code=''.join(buf)
        if code in inv: out.append(inv[code])
    return ''.join(out)

# ========================= 의사라벨 생성 =========================
def decode_with_conf_s1(ckpt_path: str, audio_path: str, cfg: Config, bp_bw_hz: int=150):
    """S1 토큰 기준 confidence(프레임 max 평균; blank 제외)와 토큰열→문자 변환 텍스트 반환"""
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt=torch.load(ckpt_path, map_location=device); vocab=Vocab(ckpt.get("vocab", S1_TOKENS))
    n_class=len(vocab.chars)+1
    model=CWCTC(cfg.n_mels,n_class,cfg.hidden,cfg.cnn_ch).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    wav=load_audio_any(audio_path, sr=cfg.sr)
    wav=pre_emphasis(wav, cfg.pre_emph)
    x_env,_ = normalize_pipeline(wav, cfg)
    mel=wav_to_logmel(x_env, cfg.sr, cfg.n_mels, cfg.hop_ms, cfg.win_ms, cfg.fmin, cfg.fmax)
    with torch.no_grad():
        x=torch.from_numpy(mel).unsqueeze(0).unsqueeze(1).to(device)
        logits=model(x)
        prob=logits.softmax(-1)[0].cpu()
        pred_ids=prob.argmax(-1).numpy().tolist()
        blank_id=len(vocab.chars)
        mask=np.array([i!=blank_id for i in pred_ids])
        conf=float(prob.max(-1).values.numpy()[mask].mean()) if mask.any() else 0.0
        tokens=vocab.ids2seq_ctc(pred_ids)
    text = s1_tokens_to_text(tokens)
    return text, conf

def make_pseudo_csv(cfg: Config, ckpt: str, in_dir: str, out_csv: str,
                    conf_th: float=0.75, agree_only: bool=False):
    files=[]
    for ext in ("*.mp3","*.wav","*.flac","*.ogg"):
        files += glob.glob(os.path.join(in_dir, ext))
    files.sort()
    kept=0; rows=[]
    print(f"[make_pseudo] scan {len(files)} files ...")
    for i,path in enumerate(files,1):
        try:
            text, conf = decode_with_conf_s1(ckpt, path, cfg)
            if not text or conf < conf_th:
                continue
            if agree_only:
                wav=load_audio_any(path, sr=cfg.sr)
                dsp_txt=dsp_morse_decode(wav, cfg.sr, fmin=cfg.fmin, fmax=cfg.fmax)
                if (dsp_txt or "").strip() != text.strip():
                    continue
            rows.append([path, text]); kept+=1
        except Exception:
            pass
        if i%20==0:
            print(f"  processed {i}/{len(files)} kept={kept}")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["path","text"]); w.writerows(rows)
    print(f"[make_pseudo] kept {kept} / {len(files)} → {out_csv}")

# ========================= CLI/메인 =========================
def parse_args():
    ap=argparse.ArgumentParser(description="CW(Morse) 정규화 + 커리큘럼(Stage1/2) 파이프라인")
    # 특징/오디오
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--hop_ms", type=float, default=10.0)
    ap.add_argument("--win_ms", type=float, default=25.0)
    ap.add_argument("--fmin", type=int, default=40)
    ap.add_argument("--fmax", type=int, default=4000)
    ap.add_argument("--pre_emph", type=float, default=0.0)
    # 정규화
    ap.add_argument("--normalize", action="store_true", help="정규화 파이프라인 사용")
    ap.add_argument("--heterodyne", action="store_true", help="f0 하향변환(포락선)")
    ap.add_argument("--bp_pre", action="store_true", help="좁은 대역통과 기반 포락선")
    ap.add_argument("--bp_bw_hz", type=int, default=150)
    ap.add_argument("--tnorm", action="store_true", help="속도(dit) 정규화")
    # 커리큘럼
    ap.add_argument("--stage", choices=["s1","s2"], default="s1", help="s1=토큰 CTC, s2=토큰→문자 후처리")
    # 합성 학습
    ap.add_argument("--train_synth", action="store_true")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--save_ckpt", type=str, default="cw_s1.pt")
    # 파인튜닝
    ap.add_argument("--load_ckpt", type=str, help="체크포인트 경로 (s1)")
    ap.add_argument("--finetune_csv", type=str, help="라벨 CSV path,text (문자 라벨)")
    ap.add_argument("--finetune_epochs", type=int, default=5)
    ap.add_argument("--finetune_lr", type=float, default=1e-4)
    ap.add_argument("--finetune_bs", type=int, default=8)
    ap.add_argument("--freeze_cnn", action="store_true")
    ap.add_argument("--freeze_rnn", action="store_true")
    # 추론
    ap.add_argument("--audio_in", type=str, help="추론 오디오(mp3/wav)")
    ap.add_argument("--out_txt", type=str, default=None)
    # 의사라벨
    ap.add_argument("--make_pseudo", action="store_true")
    ap.add_argument("--in_dir", type=str)
    ap.add_argument("--out_csv", type=str, default="pseudo.csv")
    ap.add_argument("--conf_th", type=float, default=0.75)
    ap.add_argument("--agree_only", action="store_true")
    return ap.parse_args()

def main():
    args=parse_args()
    cfg=Config(sr=args.sr, n_mels=args.n_mels, hop_ms=args.hop_ms, win_ms=args.win_ms,
               fmin=args.fmin, fmax=args.fmax, pre_emph=args.pre_emph,
               normalize=args.normalize, heterodyne=args.heterodyne,
               bp_pre=args.bp_pre or (not args.heterodyne and args.normalize),
               bp_bw_hz=args.bp_bw_hz, tnorm=args.tnorm,
               batch_size=args.batch_size, epochs=args.epochs)

    did=False

    # 1) 합성 학습 (Stage1)
    if args.train_synth:
        if args.stage!="s1":
            print("[WARN] 합성학습은 Stage1 토큰 CTC 기준으로 동작합니다. --stage s1 권장.")
        train_s1(cfg, args.save_ckpt); did=True

    # 2) 파인튜닝 (CSV 문자 라벨 → Stage1 토큰으로 변환하여 학습)
    if args.load_ckpt and args.finetune_csv:
        finetune_s1(cfg, args.load_ckpt, args.finetune_csv,
                    save_ckpt=args.save_ckpt or "cw_s1_ft.pt",
                    epochs=args.finetune_epochs, lr=args.finetune_lr,
                    batch_size=args.finetune_bs,
                    freeze_cnn=args.freeze_cnn, freeze_rnn=args.freeze_rnn)
        did=True

    # 3) 추론 (s1: 토큰열 출력, s2: 문자 출력)
    if args.load_ckpt and args.audio_in:
        tokens = infer_s1_to_tokens(cfg, args.load_ckpt, args.audio_in)
        if args.stage=="s2":
            text = s1_tokens_to_text(tokens)
        else:
            text = ' '.join(tokens)
        print("----- DECODED -----")
        print(text)
        print("-------------------")
        if args.out_txt:
            with open(args.out_txt,"w",encoding="utf-8") as f: f.write(text)
            print(f"💾 Saved: {args.out_txt}")
        # DSP 참고 출력
        try:
            wav=load_audio_any(args.audio_in, sr=cfg.sr)
            dsp_txt=dsp_morse_decode(wav, cfg.sr, fmin=cfg.fmin, fmax=cfg.fmax)
            if dsp_txt:
                print("----- DSP BASELINE -----"); print(dsp_txt); print("------------------------")
        except Exception:
            pass
        did=True

    # 4) 의사라벨 CSV 생성 (문자 라벨 생성)
    if args.make_pseudo:
        if not args.load_ckpt:
            print("[ERROR] --make_pseudo 는 --load_ckpt(베이스 s1 모델)가 필요합니다.")
        elif not args.in_dir:
            print("[ERROR] --make_pseudo 는 --in_dir(오디오 폴더)가 필요합니다.")
        else:
            make_pseudo_csv(cfg, args.load_ckpt, args.in_dir, args.out_csv,
                            conf_th=args.conf_th, agree_only=args.agree_only)
        did=True

    if not did:
        print("Nothing to do.\n"
              "- 합성 학습(Stage1): --train_synth --normalize --bp_pre --epochs 10 --save_ckpt cw_s1.pt\n"
              "- 파인튜닝(Stage1):  --load_ckpt cw_s1.pt --finetune_csv data.csv --normalize --bp_pre --save_ckpt cw_s1_ft.pt\n"
              "- 추론(토큰):        --load_ckpt cw_s1_ft.pt --audio_in input.mp3 --normalize --bp_pre --stage s1 --out_txt out.txt\n"
              "- 추론(문자 치환):   --load_ckpt cw_s1_ft.pt --audio_in input.mp3 --normalize --bp_pre --stage s2 --out_txt out.txt\n"
              "- 의사라벨:          --make_pseudo --load_ckpt cw_s1.pt --in_dir FOLDER --out_csv pseudo.csv --normalize --bp_pre")

if __name__ == "__main__":
    main()

# ============================ 사용 예시 ============================
# (PowerShell)
#
# 0) VRAM 부족 시
#    $env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
#
# 1) 합성 데이터로 Stage1(토큰) 베이스 학습 + 정규화 사용
#    python DL.test.py --train_synth --normalize --bp_pre --epochs 12 --batch_size 8 --save_ckpt cw_s1.pt
#
# 2) (선택) 의사라벨 생성 → CSV (문자 라벨 자동)
#    python DL.test.py --make_pseudo --load_ckpt cw_s1.pt --in_dir C:\data\cw200 --out_csv C:\data\cw200\pseudo.csv --normalize --bp_pre --conf_th 0.75 --agree_only
#
# 3) Stage1 파인튜닝(문자 라벨 CSV를 토큰으로 내부 변환)
#    python DL.test.py --load_ckpt cw_s1.pt --finetune_csv C:\data\cw200\pseudo.csv --normalize --bp_pre --finetune_epochs 3 --finetune_lr 1e-4 --finetune_bs 8 --save_ckpt cw_s1_ft.pt
#
# 4) 추론
#   (a) 토큰열만 보고 싶을 때:
#    python DL.test.py --load_ckpt cw_s1_ft.pt --audio_in feed.mp3 --normalize --bp_pre --stage s1 --out_txt out_tokens.txt
#   (b) 최종 문자로 보고 싶을 때(권장):
#    python DL.test.py --load_ckpt cw_s1_ft.pt --audio_in feed.mp3 --normalize --bp_pre --stage s2 --out_txt out.txt
#
# 5) 옵션 메모
#   --heterodyne  : f0 하향변환 포락선(정확하지만 연산 조금↑)
#   --bp_pre      : 좁은 대역통과 + 포락선(빠르고 튼튼, 기본 추천)
#   --tnorm       : dit 길이 표준화(속도 분산 클 때 켜면 좋음)
#
