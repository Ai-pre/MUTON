# Datasets

MUTON uses both Korean multimodal samples and MELD-based auxiliary samples. The dataset design changed between P-project and Graduation Project 2 because the model input format changed.

## Korean Multimodal Samples

The Korean dataset is built around service-level utterance units.

Main components:

- face crop or representative face frame
- utterance-level audio
- Korean transcript
- emotion or context label when available
- Korean summary target for the model path

Processing flow:

1. Track the target speaker in the source video.
2. Extract and align the face region.
3. Convert the audio track to `16kHz` mono PCM.
4. Segment speech into utterance-level units.
5. Align face, audio, and transcript data.
6. Build summary targets for Korean experiments.

## MELD Auxiliary Samples

MELD is an English multimodal dialogue dataset, but it is useful for this project because facial expression, speaking tone, and conversational turn changes are not limited to one language. The spoken text is English, but the visible emotional reaction and tone can still support multimodal learning after translation and alignment.

MELD is not used as raw foreign-language data. It is reconstructed before use:

1. Match MELD CSV rows with video clips.
2. Translate utterance text into Korean.
3. Extract representative frames and utterance audio.
4. Use translated text and emotion metadata to create pseudo-summary targets.
5. Export the result into multimodal samples that match the Qwen workflow.

## P-project Format

P-project used feature-oriented samples. Face, audio, and text encoders produced vectors, and those vectors were passed into a custom fusion Transformer.

This format was useful for proving the custom architecture, but it compressed each modality before summary generation.

## Graduation Project 2 Format

Graduation Project 2 adds a generation-oriented dataset format. Image, audio, and text are prepared in JSONL-style message samples so that Qwen2.5-Omni can use multimodal context more directly.

Related scripts:

```text
scripts/build_rich_ko_dataset.py
scripts/build_rich_meld_dataset.py
scripts/export_qwen_omni_ko_dataset.py
scripts/export_qwen_omni_meld_dataset.py
src/qwen_omni_dataset.py
```
