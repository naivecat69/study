#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cw_dl_morse_e2e.py  (통합판)
- 합성 CW(모스) 학습 (--train_synth)
- 라벨 CSV 파인튜닝 (--load_ckpt --finetune_csv)
- 단일 파일 추론 (--load_ckpt --audio_in)
- 폴더 일괄 의사라벨링 → CSV 생성 (--make_pseudo --in_dir --out_csv)

필수 패키지:
  pip install torch torchaudio librosa soundfile numpy scipy
"""

import os, math, random, argparse, string, glob, csv
from dataclasses import dataclass, replace
from typing import List
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, lfilter, welch

# ---------------------------------------------------------------------
# 0) 공통 설정
# ---------------------------------------------------------------------
SEED = 2025
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

@dataclass
class Config:
    # 오디오/특징
    sr: int = 8000
    n_mels: int = 64
    hop_ms: float = 10.0
    win_ms: float = 25.0
    fmin: int = 40
    fmax: int = 4000
    pre_emph: float = 0.0
    bp_pre: bool = False
    bp_bw_hz: int = 200
    # 합성
    synth_wpm_min: int = 14
    synth_wpm_max: int = 28
    synth_f0_min: int = 300
    synth_f0_max: int = 1200
    atk_ms: float = 5.0
    dcy_ms: float = 5.0
    snr_db_min: float = 20.0
    snr_db_max: float = 40.0
    # 학습
    batch_size: int = 16
    epochs: int = 10
    lr: float = 3e-4
    # 모델
    hidden: int = 128
    cnn_ch: int = 64

# ---------------------------------------------------------------------
# 1) 모스 테이블 & 문자집합
# ---------------------------------------------------------------------
MORSE = {
    'A':'.-','B':'-...','C':'-.-.','D':'-..','E':'.','F':'..-.','G':'--.','H':'....','I':'..',
    'J':'.---','K':'-.-','L':'.-..','M':'--','N':'-.','O':'---','P':'.--.','Q':'--.-','R':'.-.',
    'S':'...','T':'-','U':'..-','V':'...-','W':'.--','X':'-..-','Y':'-.--','Z':'--..',
    '0':'-----','1':'.----','2':'..---','3':'...--','4':'....-','5':'.....',
    '6':'-....','7':'--...','8':'---..','9':'----.',
    '.':'.-.-.-', ',':'--..--', '?':'..--..', '/':'-..-.', '-':'-....-',
    ' ':' '
}
VOCAB_CHARS = list(string.ascii_uppercase + string.digits + " .,?/-")  # 공백 포함

# ---------------------------------------------------------------------
# 2) 합성/전처리/특징
# ---------------------------------------------------------------------
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

def estimate_peak_hz(x: np.ndarray, sr: int, fmin=80, fmax=4000) -> float:
    f,P = welch(x, fs=sr, nperseg=4096)
    m = (f>=fmin)&(f<=fmax)
    return float(f[m][np.argmax(P[m])]) if np.any(m) else 600.0

def butter_bandpass(lowcut, highcut, fs, order=4):
    return butter(order, [lowcut/(fs/2), highcut/(fs/2)], btype='band')

def apply_bandpass(x: np.ndarray, sr: int, center: float, bw_hz: float) -> np.ndarray:
    low = max(20.0, center-bw_hz/2)
    high = min(sr/2-50.0, center+bw_hz/2)
    if low>=high: return x
    b,a = butter_bandpass(low, high, sr, order=4)
    return lfilter(b,a,x).astype(np.float32)

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

# ---------------------------------------------------------------------
# 3) Vocab & 모델
# ---------------------------------------------------------------------
class Vocab:
    def __init__(self, chars: List[str]):
        self.chars = chars[:]
        self.blank_id = len(self.chars)
        self.stoi = {c:i for i,c in enumerate(self.chars)}
        self.itos = {i:c for i,c in enumerate(self.chars)}
        self.itos[self.blank_id] = ''
    def text2ids(self, text: str) -> List[int]:
        return [self.stoi[c] for c in text.upper() if c in self.stoi]
    def ids2text_ctc(self, ids: List[int]) -> str:
        out, prev = [], None
        for i in ids:
            if i==self.blank_id: prev=i; continue
            if i!=prev: out.append(self.itos.get(i,''))
            prev=i
        return ''.join(out)

def normalize_text_for_vocab(text: str, vocab_chars: List[str]) -> str:
    allow=set(vocab_chars); cleaned=[]; prev_space=False
    for ch in text.upper():
        if ch in allow:
            if ch==' ':
                if not prev_space: cleaned.append(' ')
                prev_space=True
            else:
                cleaned.append(ch); prev_space=False
    return ''.join(cleaned).strip()

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

# ---------------------------------------------------------------------
# 4) 데이터셋
# ---------------------------------------------------------------------
class SynthDataset(Dataset):
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
        if len(wav)>0 and random.random()<0.5:
            drift = 1.0 + np.linspace(0, random.uniform(-0.002,0.002), len(wav))
            wav = (wav*drift).astype(np.float32)
        snr = random.uniform(self.cfg.snr_db_min, self.cfg.snr_db_max)
        wav = add_awgn(wav, snr)
        wav = pre_emphasis(wav, self.cfg.pre_emph)
        mel = wav_to_logmel(wav, self.cfg.sr, self.cfg.n_mels, self.cfg.hop_ms, self.cfg.win_ms,
                            self.cfg.fmin, self.cfg.fmax)
        target = self.vocab.text2ids(txt)
        return mel, np.array(target, np.int32), txt

class LabeledAudioDataset(Dataset):
    def __init__(self, csv_path: str, cfg: Config, vocab: Vocab, use_bp_pre=False, shuffle=True):
        self.cfg=cfg; self.vocab=vocab; self.use_bp_pre=use_bp_pre
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
        if self.use_bp_pre or self.cfg.bp_pre:
            pk = estimate_peak_hz(wav, self.cfg.sr, fmin=self.cfg.fmin, fmax=min(self.cfg.fmax, self.cfg.sr//2-100))
            wav = apply_bandpass(wav, self.cfg.sr, center=pk, bw_hz=self.cfg.bp_bw_hz)
        wav = pre_emphasis(wav, self.cfg.pre_emph)
        mel = wav_to_logmel(wav, self.cfg.sr, self.cfg.n_mels, self.cfg.hop_ms, self.cfg.win_ms,
                            self.cfg.fmin, self.cfg.fmax)
        txt = normalize_text_for_vocab(raw, self.vocab.chars)
        target = self.vocab.text2ids(txt)
        return mel, np.array(target, np.int32), txt

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

# ---------------------------------------------------------------------
# 5) 학습/파인튜닝/추론
# ---------------------------------------------------------------------
def train_synth(cfg: Config, save_ckpt: str):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab=Vocab(VOCAB_CHARS); n_class=len(vocab.chars)+1
    model=CWCTC(cfg.n_mels,n_class,cfg.hidden,cfg.cnn_ch).to(device)
    ds=SynthDataset(cfg,vocab,n_items=2000)
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
        print(f"[Epoch {ep}/{cfg.epochs}] CTC loss={tot/max(1,len(dl)):.4f}")
    os.makedirs(os.path.dirname(save_ckpt) or ".", exist_ok=True)
    torch.save({"model":model.state_dict(),"cfg":cfg.__dict__,"vocab":vocab.chars}, save_ckpt)
    print(f"✅ Saved checkpoint: {save_ckpt}")

def finetune_supervised(cfg: Config, ckpt_in: str, csv_path: str, save_ckpt: str,
                        epochs=5, lr=1e-4, batch_size=8, freeze_cnn=False, freeze_rnn=False, bp_pre_override=False):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt=torch.load(ckpt_in, map_location=device); vocab=Vocab(ckpt["vocab"])
    n_class=len(vocab.chars)+1
    model=CWCTC(cfg.n_mels,n_class,cfg.hidden,cfg.cnn_ch).to(device)
    model.load_state_dict(ckpt["model"])
    if freeze_cnn: [setattr(p,'requires_grad',False) for p in model.cnn.parameters()]
    if freeze_rnn: [setattr(p,'requires_grad',False) for p in model.rnn.parameters()]
    ds=LabeledAudioDataset(csv_path, cfg, vocab, use_bp_pre=bp_pre_override)
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
        avg=tot/max(1,len(dl)); print(f"[FT {ep}/{epochs}] loss={avg:.4f}")
        if avg<best:
            best=avg
            torch.save({"model":model.state_dict(),"cfg":cfg.__dict__,"vocab":vocab.chars}, save_ckpt)
            print(f"  ↳ ✅ improved; saved: {save_ckpt}")

def dsp_morse_decode(wav: np.ndarray, sr: int, fmin=100, fmax=3000):
    pk=estimate_peak_hz(wav,sr,fmin=fmin,fmax=fmax)
    x=apply_bandpass(wav,sr,center=pk,bw_hz=120); x=np.abs(x)
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

def infer_file(cfg: Config, ckpt_path: str, audio_path: str, out_txt: str=None):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt=torch.load(ckpt_path, map_location=device); vocab=Vocab(ckpt["vocab"])
    n_class=len(vocab.chars)+1
    model=CWCTC(cfg.n_mels,n_class,cfg.hidden,cfg.cnn_ch).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    wav=load_audio_any(audio_path, sr=cfg.sr)
    pk0=estimate_peak_hz(wav,cfg.sr,fmin=cfg.fmin,fmax=min(cfg.fmax,cfg.sr//2-100))
    if cfg.bp_pre:
        wav=apply_bandpass(wav,cfg.sr,center=pk0,bw_hz=cfg.bp_bw_hz)
    wav=pre_emphasis(wav,cfg.pre_emph)
    mx=np.max(np.abs(wav))+1e-9; wav=(wav/mx).astype(np.float32)

    mel=wav_to_logmel(wav,cfg.sr,cfg.n_mels,cfg.hop_ms,cfg.win_ms,cfg.fmin,cfg.fmax)
    with torch.no_grad():
        x=torch.from_numpy(mel).unsqueeze(0).unsqueeze(1).to(device)
        logits=model(x)
        pred=logits.argmax(-1).squeeze(0).tolist()
        text=vocab.ids2text_ctc(pred).strip()

    print("----- DECODED TEXT -----"); print(text); print("------------------------")
    if out_txt:
        with open(out_txt,"w",encoding="utf-8") as f: f.write(text)
        print(f"💾 Saved: {out_txt}")
    # DSP 백업도 출력
    try:
        dsp_txt=dsp_morse_decode(wav, cfg.sr, fmin=cfg.fmin, fmax=cfg.fmax)
        if dsp_txt:
            print("----- DSP BASELINE -----"); print(dsp_txt); print("------------------------")
    except Exception as e:
        pass
    return text

# ---------------------------------------------------------------------
# 6) ★ 의사라벨 생성 (폴더 → pseudo.csv)
# ---------------------------------------------------------------------
def decode_with_conf(ckpt_path: str, audio_path: str, cfg: Config, bp_bw_hz: int=150):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt=torch.load(ckpt_path, map_location=device); vocab=Vocab(ckpt["vocab"])
    n_class=len(vocab.chars)+1
    model=CWCTC(cfg.n_mels,n_class,cfg.hidden,cfg.cnn_ch).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    wav=load_audio_any(audio_path, sr=cfg.sr)
    pk=estimate_peak_hz(wav,cfg.sr,fmin=cfg.fmin,fmax=min(cfg.fmax,cfg.sr//2-100))
    wav=apply_bandpass(wav,cfg.sr,center=pk,bw_hz=bp_bw_hz)
    wav=pre_emphasis(wav,cfg.pre_emph)
    mx=np.max(np.abs(wav))+1e-9; wav=(wav/mx).astype(np.float32)

    mel=wav_to_logmel(wav,cfg.sr,cfg.n_mels,cfg.hop_ms,cfg.win_ms,cfg.fmin,cfg.fmax)
    with torch.no_grad():
        x=torch.from_numpy(mel).unsqueeze(0).unsqueeze(1).to(device)
        logits=model(x)
        prob=logits.softmax(-1)[0].cpu()           # (T,C)
        pred_ids=prob.argmax(-1).numpy().tolist()
        blank_id=len(vocab.chars)
        mask=np.array([i!=blank_id for i in pred_ids])
        conf=float(prob.max(-1).values.numpy()[mask].mean()) if mask.any() else 0.0
        text=vocab.ids2text_ctc(pred_ids).strip()
    return text, conf

def make_pseudo_csv(cfg: Config, ckpt: str, in_dir: str, out_csv: str,
                    conf_th: float=0.75, agree_only: bool=False, bp_bw_hz: int=150):
    files=[]
    for ext in ("*.mp3","*.wav","*.flac","*.ogg"):
        files += glob.glob(os.path.join(in_dir, ext))
    files.sort()
    kept=0; rows=[]
    print(f"[make_pseudo] scan {len(files)} files ...")
    for i,path in enumerate(files,1):
        try:
            text, conf = decode_with_conf(ckpt, path, cfg, bp_bw_hz=bp_bw_hz)
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

# ---------------------------------------------------------------------
# 7) CLI
# ---------------------------------------------------------------------
def parse_args():
    ap=argparse.ArgumentParser(description="CW(Morse) CTC 학습/파인튜닝/추론/의사라벨")
    # 특징/오디오
    ap.add_argument("--sr", type=int, default=8000)
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--hop_ms", type=float, default=10.0)
    ap.add_argument("--win_ms", type=float, default=25.0)
    ap.add_argument("--fmin", type=int, default=40)
    ap.add_argument("--fmax", type=int, default=4000)
    ap.add_argument("--pre_emph", type=float, default=0.0)
    ap.add_argument("--bp_pre", action="store_true")
    ap.add_argument("--bp_bw_hz", type=int, default=200)
    # 합성 학습
    ap.add_argument("--train_synth", action="store_true")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--save_ckpt", type=str, default="cw_ctc.pt")
    # 파인튜닝
    ap.add_argument("--load_ckpt", type=str, help="체크포인트 경로")
    ap.add_argument("--finetune_csv", type=str, help="라벨 CSV path,text")
    ap.add_argument("--finetune_epochs", type=int, default=5)
    ap.add_argument("--finetune_lr", type=float, default=1e-4)
    ap.add_argument("--finetune_bs", type=int, default=8)
    ap.add_argument("--freeze_cnn", action="store_true")
    ap.add_argument("--freeze_rnn", action="store_true")
    ap.add_argument("--bp_pre_override", action="store_true")
    # 추론
    ap.add_argument("--audio_in", type=str, help="추론 오디오(mp3/wav)")
    ap.add_argument("--out_txt", type=str, default=None)
    # ★ 의사라벨
    ap.add_argument("--make_pseudo", action="store_true", help="폴더 일괄 의사라벨 CSV 생성")
    ap.add_argument("--in_dir", type=str, help="의사라벨 입력 폴더")
    ap.add_argument("--out_csv", type=str, default="pseudo.csv")
    ap.add_argument("--conf_th", type=float, default=0.75)
    ap.add_argument("--agree_only", action="store_true", help="DL/DSP 일치한 샘플만 채택")
    ap.add_argument("--pseudo_bw_hz", type=int, default=150, help="의사라벨용 BPF 대역폭")
    return ap.parse_args()

def main():
    args=parse_args()
    cfg=Config(sr=args.sr, n_mels=args.n_mels, hop_ms=args.hop_ms, win_ms=args.win_ms,
               fmin=args.fmin, fmax=args.fmax, pre_emph=args.pre_emph,
               bp_pre=args.bp_pre, bp_bw_hz=args.bp_bw_hz,
               batch_size=args.batch_size, epochs=args.epochs)

    did=False

    # 1) 합성학습
    if args.train_synth:
        train_synth(cfg, args.save_ckpt); did=True

    # 2) 파인튜닝
    if args.load_ckpt and args.finetune_csv:
        finetune_supervised(cfg, args.load_ckpt, args.finetune_csv,
                            save_ckpt=args.save_ckpt or "cw_ctc_finetuned.pt",
                            epochs=args.finetune_epochs, lr=args.finetune_lr,
                            batch_size=args.finetune_bs,
                            freeze_cnn=args.freeze_cnn, freeze_rnn=args.freeze_rnn,
                            bp_pre_override=args.bp_pre_override)
        did=True

    # 3) 추론
    if args.load_ckpt and args.audio_in:
        infer_file(cfg, args.load_ckpt, args.audio_in, args.out_txt); did=True

    # 4) ★ 의사라벨 CSV 생성
    if args.make_pseudo:
        if not args.load_ckpt:
            print("[ERROR] --make_pseudo 는 --load_ckpt(베이스 모델)가 필요합니다.")
        elif not args.in_dir:
            print("[ERROR] --make_pseudo 는 --in_dir(오디오 폴더)가 필요합니다.")
        else:
            cfg2 = replace(cfg, bp_pre=True)  # 의사라벨링은 협대역 BPF 기본 적용 권장
            make_pseudo_csv(cfg2, args.load_ckpt, args.in_dir, args.out_csv,
                            conf_th=args.conf_th, agree_only=args.agree_only,
                            bp_bw_hz=args.pseudo_bw_hz)
        did=True

    if not did:
        print("Nothing to do.\n"
              "- 합성 학습: --train_synth --epochs 10 --save_ckpt base.pt\n"
              "- 파인튜닝:  --load_ckpt base.pt --finetune_csv data.csv --save_ckpt ft.pt\n"
              "- 추론:      --load_ckpt model.pt --audio_in input.mp3 --out_txt out.txt\n"
              "- 의사라벨:  --make_pseudo --load_ckpt base.pt --in_dir FOLDER --out_csv pseudo.csv")

if __name__ == "__main__":
    main()

# ============================ 사용 예시 ============================
# (PowerShell 기준, 파일명이 DL.test.py라면 그걸로 실행)
#
# 0) (선택) VRAM 부족 시
#    $env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
#
# 1) 합성 데이터로 베이스 모델 만들기
#    python DL.test.py --train_synth --epochs 12 --batch_size 8 --save_ckpt base.pt
#
# 2) 폴더 일괄 의사라벨 생성 (confidence 0.75 이상, DSP 일치 샘플만)
#    python DL.test.py `
#       --make_pseudo `
#       --load_ckpt base.pt `
#       --in_dir C:\data\cw200 `
#       --out_csv C:\data\cw200\pseudo.csv `
#       --sr 16000 `
#       --conf_th 0.75 `
#       --agree_only `
#       --pseudo_bw_hz 150
#
# 3) 의사라벨 CSV로 파인튜닝
#    python DL.test.py `
#       --load_ckpt base.pt `
#       --finetune_csv C:\data\cw200\pseudo.csv `
#       --finetune_epochs 3 `
#       --finetune_lr 1e-4 `
#       --finetune_bs 8 `
#       --bp_pre_override `
#       --save_ckpt ft.pt
#
# 4) 추론
#    python DL.test.py --load_ckpt ft.pt --audio_in C:\data\cw200\example.wav --bp_pre --bp_bw_hz 150 --sr 16000 --out_txt out.txt
#
# 5) 메모리 부족하면 (모델 축소)
#    -batch_size 4 또는 2로 줄이기
#    Config(hidden=64, cnn_ch=32)로 축소 후 재학습
