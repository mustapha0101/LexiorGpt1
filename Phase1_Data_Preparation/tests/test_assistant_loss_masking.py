from assistant_loss import encode_messages_assistant_only, parse_strict_chatml, render_strict_chatml


class CharTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) for char in text]}


def test_only_assistant_bodies_have_labels_token_by_token():
    tokenizer = CharTokenizer()
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
        {"role": "assistant", "content": "A1<tool_call>{}</tool_call>"},
        {"role": "tool", "content": "LONG TOOL RESPONSE"},
        {"role": "assistant", "content": "A2 final"},
    ]
    encoded = encode_messages_assistant_only(tokenizer, messages)
    cursor = 0
    for message in messages:
        header = f"<|im_start|>{message['role']}\n"
        body = message["content"] + "<|im_end|>\n"
        header_len, body_len = len(header), len(body)
        assert encoded["labels"][cursor:cursor + header_len] == [-100] * header_len
        cursor += header_len
        body_labels = encoded["labels"][cursor:cursor + body_len]
        if message["role"] == "assistant":
            assert body_labels == [ord(char) for char in body]
        else:
            assert body_labels == [-100] * body_len
        cursor += body_len
    assert cursor == len(encoded["input_ids"])


def test_strict_chatml_round_trip_without_qwen_injection():
    messages = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    text = render_strict_chatml(messages)
    assert "You are Qwen" not in text
    assert parse_strict_chatml(text) == messages
