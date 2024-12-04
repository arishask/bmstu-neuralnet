import torch
from collections import deque
from transformers import AutoTokenizer, AutoModelWithLMHead


class DialogBotRuGPTSmall:
    def __init__(self, pretrained_model='tinkoff-ai/ruDialoGPT-small'):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model = AutoModelWithLMHead.from_pretrained(pretrained_model)
        self.context = deque([], maxlen=4)  # 2 inputs, 2 answers
        self.rus_alphabet = "йцукенгшщзхъфывапролджэячсмитьбюёЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮЁ"
        self.en_alphabet = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM"
        self.symbols = "0123456789., !?\':;@#$&*()-=+"

    def predict(self, text):
        if len(self.context) == self.context.maxlen:
            _ = self.context.popleft()
            _ = self.context.popleft()
        self.context.append(text)
        input_text = [f"@@ПЕРВЫЙ@@ {t}" if not i % 2 else f"@@ВТОРОЙ@@ {t}"
                      for i, t in enumerate(self.context)]
        input_text = " ".join(input_text) + " @@ВТОРОЙ@@ "

        tokenized_text = self.tokenizer(input_text, return_tensors='pt')
        with torch.no_grad():
            generated_token_ids = self.model.generate(
                **tokenized_text,
                top_k=10,
                top_p=0.95,
                num_beams=3,
                num_return_sequences=1,
                do_sample=True,
                no_repeat_ngram_size=2,
                temperature=1.2,
                repetition_penalty=1.2,
                length_penalty=1.0,
                eos_token_id=50257,
                max_new_tokens=40
            )

            context_with_response = [self.tokenizer.decode(sample_token_ids) for sample_token_ids in
                                     generated_token_ids]
            answer = context_with_response[0].split("@@ВТОРОЙ@@")[-1].replace("@@ПЕРВЫЙ@@", "")
            filtered_answer = [letter for letter in answer if letter in self.rus_alphabet
                               or letter in self.en_alphabet or letter in self.symbols]
            filtered_answer = "".join(filtered_answer).strip()
            self.context.append(filtered_answer)

        return answer
