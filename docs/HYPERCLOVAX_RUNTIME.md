# HyperCLOVA Runtime Runbook

## Recommended Deployment Split

- STT: `OpenAI whisper-1`
- summary/reasoning: `HyperCLOVAX-SEED-Omni-8B` through OmniServe
- mobile transport: FastAPI + Cloudflare tunnel + `backend_url.json`

This branch keeps the STT path from `server_main` and swaps only the multimodal summary backend to HyperCLOVA.

## Server Start

```bash
cd ~/MUTON_cpy
git fetch origin
git checkout hyperclovax-omni-exp
git reset --hard origin/hyperclovax-omni-exp

source ~/miniconda3/etc/profile.d/conda.sh
conda activate muton

export OPENAI_API_KEY=YOUR_KEY
export MUTON_HC_BASE_URL=http://127.0.0.1:8000/b/v1
export MUTON_HC_MODEL_NAME=track_b_model
export MUTON_HC_STT_BACKEND=openai

python scripts/run_hyperclovax_server.py
```

## OmniServe Assumption

`src/server_hyperclovax.py` assumes an OpenAI-compatible HyperCLOVA endpoint is already running.

Important environment variables:

- `MUTON_HC_BASE_URL`
- `MUTON_HC_API_KEY`
- `MUTON_HC_MODEL_NAME`
- `MUTON_HC_STT_BACKEND`
- `MUTON_HC_MAX_TOKENS`

## STT Backend Modes

### `MUTON_HC_STT_BACKEND=openai`

Use this when subtitle quality matters most.

- restores the earlier `whisper-1` API path
- uses OpenAI transcript metadata such as `avg_logprob` and `no_speech_prob`
- behaves best in noisy real-world audio

### `MUTON_HC_STT_BACKEND=whisper`

Use this when API cost matters more than subtitle quality.

- fully local
- Korean-tuned fallback model
- still uses the same utterance-level filtering helpers in `src/encoders.py`

## Sync Between STT And Summary

The HyperCLOVA server now follows the same utterance snapshot logic as `server_main`:

1. buffered audio reaches utterance-final state through VAD
2. transcript is finalized
3. that utterance waveform is staged as a `.wav`
4. the latest face image at that moment is frozen
5. `/get_fusion_analysis` uses only that committed snapshot

This avoids mixing a new transcript with an older audio buffer.

## Media Serving Model

HyperCLOVA needs public media references, so the server mounts:

```text
/media/<generated-file>
```

and stages:

- face crop JPEGs
- utterance-level WAV files

`/get_fusion_analysis` builds public URLs from the incoming request host. In practice this means the FastAPI server should be called through the same Cloudflare tunnel that HyperCLOVA can reach.

## Confidence Flow

`src/encoders.py` still computes transcript confidence.

- low-confidence subtitles are dropped
- if committed STT confidence is below `MUTON_STT_SUMMARY_MIN_CONFIDENCE`, `/get_fusion_analysis` returns `Low Confidence`
- HyperCLOVA summary is skipped in that case

Useful knobs:

```bash
export MUTON_STT_MIN_TRANSCRIPT_CONFIDENCE=0.45
export MUTON_STT_SUMMARY_MIN_CONFIDENCE=0.55
```

## Remote URL Update

The Android app should read:

```text
https://raw.githubusercontent.com/Ai-pre/MUTON/refs/heads/server/backend_url.json
```

To update the active tunnel URL:

```bash
cd ~/MUTON_server
python scripts/update_backend_url.py https://xxxxx.trycloudflare.com
git add backend_url.json
git commit -m "Update backend URL"
git push origin HEAD:server
```

Verify with cache busting:

```bash
curl "https://raw.githubusercontent.com/Ai-pre/MUTON/refs/heads/server/backend_url.json?t=$(date +%s)"
```

## Troubleshooting

### `Incoming request ended abruptly: context canceled`

This usually means the mobile client canceled the request before the origin finished responding. It is usually an STT latency or overlapping request issue, not a HyperCLOVA transport fault.

### No summary even though subtitle exists

Check whether `/get_fusion_analysis` is returning `Low Confidence`. The subtitle may have passed the display threshold while still failing the summary threshold.

### HyperCLOVA cannot see staged media

Make sure the request reaches the FastAPI server through the Cloudflare tunnel, not `127.0.0.1`, so the generated `/media/*` URLs are public.
