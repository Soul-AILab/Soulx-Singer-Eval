import string
import zhconv
import jiwer
from zhon import hanzi
from faster_whisper import WhisperModel
from funasr import AutoModel


class ASRPipeline:

    def __init__(self, lang) -> None:
        self.lang = lang
        if self.lang == 'en':
            self.asr_model = WhisperModel("large-v3", device="cuda")
        if self.lang == 'zh':
            self.asr_model = AutoModel(model="paraformer-zh", disable_update=True, device="cuda")


    def clean_text_en(self, text):
        punctuation_str = string.punctuation
        for i in punctuation_str:
            text = text.replace(i, '')
        text = text.lower()
        return text


    def clean_text_zh(self, text):
        punctuation_str = hanzi.punctuation + " "
        for i in punctuation_str:
            text = text.replace(i, '')
        return text


    def infer_en(self, wav):
        segments, info = self.asr_model.transcribe(
            wav,
            language="en",
            # beam_size=5,
            # temperature=0,
        )
        hyp_text = ""
        for segment in segments:
            hyp_text += segment.text
        hyp_text = hyp_text.rstrip().strip()
        return hyp_text


    def infer_zh(self, wav):
        res = self.asr_model.generate(input=wav, batch_size_s=300)
        hyp_text = res[0]["text"]
        hyp_text = zhconv.convert(hyp_text, 'zh-cn')
        return hyp_text


    def get_wer(self, ref_text, hyp_text, mode="wer"):
        if self.lang == 'en':
            ref_text = self.clean_text_en(ref_text)
            hyp_text = self.clean_text_en(hyp_text)
        elif self.lang == 'zh':
            ref_text = self.clean_text_zh(ref_text)
            hyp_text = self.clean_text_zh(hyp_text)

        if self.lang == 'en' and mode == "wer":
            ref_formatted = " ".join(ref_text.split())
            hyp_formatted = " ".join(hyp_text.split())
        else:
            ref_formatted = " ".join(list(ref_text))
            hyp_formatted = " ".join(list(hyp_text))

        output = jiwer.process_words(ref_formatted, hyp_formatted)

        return {
            "ref": ref_text,
            "hyp": hyp_text,
            "wer": output.wer,
            "del": output.deletions,
            "ins": output.insertions,
            "sub": output.substitutions
        }

        

