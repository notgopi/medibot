import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SYSTEM_PROMPT = (
    "You are a careful medical triage assistant. "
    "Ask at most one targeted follow-up question at a time. "
    "Stop asking questions once you have enough information for a probable triage, "
    "then provide a tentative explanation with 1–3 likely conditions and generic next steps. "
    "Always include a safety disclaimer that this is not a diagnosis."
)

class Chatbot:
    def __init__(self, model_id, adapter_path="", max_new_tokens=200, temperature=0.7, top_p=0.95):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def _format_history(self, history):
        lines = [f"<|system|>{SYSTEM_PROMPT}"]
        for m in history:
            role = m["role"]
            content = m["content"]
            lines.append(f"<|{role}|>{content}")
        return "\n".join(lines) + "\n<|assistant|>"

    def respond(self, history):
        prompt = self._format_history(history)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            eos_token_id=self.tokenizer.eos_token_id
        )
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        reply = text.split("<|assistant|>")[-1].strip()
        stop_markers = ["Based on your symptoms", "Likely causes", "Possible conditions"]
        should_stop = any(marker.lower() in reply.lower() for marker in stop_markers)
        return reply, {"should_stop": should_stop}
