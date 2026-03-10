# MUTON
## Real-time Multimodal Conversation Assistant for People with Hearing Loss

MUTON은 청각장애인을 위한 **실시간 대화 보조 AI 시스템**이다.  
기존 STT 중심 자막 서비스가 전달하지 못했던 **말투, 감정, 뉘앙스, 의도**를  
음성·얼굴·텍스트를 결합한 멀티모달 추론을 통해 텍스트로 요약하여 제공한다.

본 프로젝트는 단순히 “음성을 문자로 변환”하는 문제를 넘어서,  
**사람이 대화를 이해하는 방식에 가까운 정보 통합 구조**를 시스템적으로 구현하는 것을 목표로 한다.

---

## 1. Background & Problem Definition

청각장애인은 일상적인 대화 상황에서 여전히 심각한 정보 손실을 겪는다.  
현재 널리 사용되는 STT 기반 자막 서비스는 다음과 같은 구조적 한계를 가진다.

- 소음 환경에서 음성 인식률 급격히 저하
- 텍스트로 변환된 발화에는 **감정, 말투, 의도 정보가 포함되지 않음**
- 대화 상대의 표정, 시선, 입모양 등 비언어적 신호를 반영하지 못함

또한 문자통역·수어통역 서비스는 인력 의존도가 높고,  
사전 예약이 필요하거나 즉각적인 사용이 어렵다는 제약이 존재한다.

특히 최근 조사에 따르면, **구화를 선호하는 청각장애인 비율이 증가**하고 있음에도  
대부분의 지원 기술은 수어 중심으로 설계되어 있다.

MUTON은 이러한 문제의식에서 출발하여,  
**구화 중심 청각장애인을 위한 실시간 멀티모달 대화 이해 시스템**을 제안한다.

---

## 2. Project Goal

MUTON의 핵심 목표는 다음과 같다.

- 음성을 단순히 텍스트로 변환하는 것을 넘어  
  **대화의 감정·톤·뉘앙스·의도를 포함한 맥락형 자막 생성**
- 실시간 모바일 환경에서 동작 가능한 **저지연 멀티모달 파이프라인**
- 음성·얼굴·텍스트를 독립적으로 처리한 뒤 **Transformer 기반으로 통합 추론**

---

## 3. System Overview

### 3.1 High-level Architecture

```

Android Device
├─ CameraX (Video Frames)
├─ AudioRecord (16kHz PCM)
│
│  (비동기 스트리밍)
▼
FastAPI Server
├─ Face Encoder
│   ├─ MediaPipe FaceMesh
│   └─ ViT Emotion Model
├─ Audio Encoder
│   ├─ VAD
│   ├─ Whisper (STT)
│   └─ WavLM
├─ Text Encoder (KLUE / KoBERT)
├─ Multimodal Fusion Transformer
│
▼
Emotion / Arousal / Valence
Context-aware Summary (GPT)

```

모바일 단말에서는 **카메라·마이크 입력을 chunk 단위로 비동기 전송**하고,  
서버에서 모든 인코딩 및 멀티모달 추론을 수행한다.

---

## 4. Face Encoder

### 4.1 Face Landmark Extraction & Alignment

- **MediaPipe FaceMesh**를 사용해 얼굴당 **478개의 랜드마크** 추출
- 양 눈 랜드마크(33, 263)를 기준으로 얼굴 기울기 계산
- 회전 변환을 적용해 얼굴을 수평으로 정렬하여 안정적인 분석 수행

### 4.2 Lip Movement-based Speaking Detection

- 입술 상·하·좌·우 랜드마크(13, 14, 61, 291)를 이용해  
  **Mouth Aspect Ratio (MAR)** 계산
- 최근 프레임들의 MAR 분산이 임계값을 초과할 경우 발화 중으로 판단
- 음성 인식 실패나 소음 환경에서도 **시각적 발화 단서**로 활용 가능

### 4.3 Facial Emotion Representation

- 얼굴 crop 이미지를 **ViT 기반 감정 분류 모델**에 입력  
  (`dima806/facial_emotions_image_detection`)
- 모델 출력:
  - 감정 logits (7 classes)
  - 감정 확률 분포
  - **마지막 hidden state의 CLS 토큰(768차원 얼굴 임베딩)**

감정 확률이 특정 임계값을 넘을 경우 해당 감정을 우선 선택하고,  
그 외의 경우 최고 확률 감정을 최종 레이블로 결정한다.

> 감정 logits은 단순 분류 결과가 아니라  
> 멀티모달 융합을 위한 **고수준 감정 표현(feature)**으로 사용된다.

---

## 5. Audio Encoder

### 5.1 Speech Segmentation & STT

- VAD(Voice Activity Detection)를 이용해 발화 구간 분리
- Whisper 기반 STT 수행
- 환각 완화를 위한 후처리:
  - no-speech 구간 제거
  - 반복 패턴 제거
  - 블랙리스트 단어 필터링

### 5.2 Audio Representation

WavLM을 이용해 다음 세 가지 음성 임베딩을 추출한다.

- **Audio Content Embedding**: 무엇을 말했는지
- **Speaker Embedding**: 화자 특성
- **Prosody Embedding**: 억양, 강세, 말하기 속도 등 말투 정보

---

## 6. Text Encoder

- Whisper STT 결과를 입력으로 **KLUE / KoBERT** 사용
- 발화 내용의 의미적 표현을 텍스트 임베딩으로 변환
- 멀티모달 융합 단계에서 의미 중심 정보 제공

---

## 7. Multimodal Fusion Transformer

### 7.1 Design Philosophy

MUTON은 멀티모달 정보를 단순히 concatenate하지 않고,  
각 모달을 **독립적인 토큰(token)**으로 취급한다.

이를 통해 다음을 학습할 수 있다.

- 같은 발화(content)라도 말투(prosody)에 따라 감정이 달라지는 현상
- 얼굴 표정과 음성 신호 간의 상호 보완 관계

### 7.2 Input Tokens

- Face Embedding (ViT, 768)
- Emotion Logits (7)
- Audio Content / Speaker / Prosody Embeddings
- Text Embedding
- CLS Token

### 7.3 Training Setup

- Projection dimension: 256
- Attention heads: 4
- Transformer blocks: 1 (경량 구조)
- Optimizer: AdamW
- Loss:
  - Emotion classification: Cross-Entropy
  - Arousal / Valence regression: MSE

---

## 8. Inference & Output

멀티모달 추론 결과로 다음을 출력한다.

- Emotion label
- Arousal / Valence
- Confidence score

해당 결과를 프롬프트로 구성해 GPT 기반 요약 문장을 생성하고,  
이를 안드로이드 앱 UI에 실시간으로 표시한다.

---

## 9. Evaluation

- Hold-out test set (200 samples)
- 감정 분류 정확도: **61.1%**
- 추가적으로 LLM 생성 요약에 대한 정성적 평가 수행

---

## 10. Key Contributions

- 얼굴·음성·텍스트를 **시간축 정렬 후 Transformer로 통합**
- 멀티모달 정보를 병렬 나열이 아닌 **상호작용 구조로 모델링**
- 구화 선호 청각장애인을 고려한 실제 사용 시나리오 중심 설계
- 실시간 처리를 고려한 비동기 서버 아키텍처

---

## 11. Future Work

- Lip-reading 기반 텍스트 보완 (MVIB-Lip)
- 화자 분리 (ECAPA / x-vector)
- 다중 화자 대화 환경 대응
- Cross-attention 디코더 구조 확장

---

## 12. Team

- Frontend: 김미리, 조혜린  
- Backend: 강지윤, 윤재상, 고유노  

---

## 13. Links

- GitHub: https://github.com/Ai-pre/MUTON  
- Demo Video: https://www.youtube.com/watch?v=dp7rl4MrPKk
# MUTON_cpy
